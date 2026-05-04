---
jupyter:
  jupytext:
    formats: md,ipynb
    main_language: python
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
---

# Train a miniGPT language model with JAX


This tutorial demonstrates how to use JAX, [Flax NNX](http://flax.readthedocs.io) and [Optax](http://optax.readthedocs.io) for language model (pre)training using data and tensor [parallelism](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization) for [Single-Program Multi-Data](https://en.wikipedia.org/wiki/Single_program,_multiple_data)).

Here, you will learn how to:

- Define the miniGPT model with Flax and JAX automatic parallelism
- Load and preprocess the dataset
- Create the loss and training step functions
- Profile for hyperparameter tuning


## Setup

We will use [Tiktoken](https://github.com/openai/tiktoken) for tokenization and [Grain](https://google-grain.readthedocs.io/en/latest/index.html) for data loading.

```python outputId="cb9e04fe-46e1-4363-b48a-9e91623abd82"
import jax
```

Get the [TinyStories dataset from Hugging Face](https://huggingface.co/datasets/roneneldan/TinyStories). We only use the training split.

```python outputId="e6eff24e-5578-4277-a0f9-24e27bd91ee0"
!wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt?download=true -O TinyStories-train.txt
```

Import the necessary modules, including JAX NumPy, Flax NNX, Optax, Grain, pandas, and Tiktoken:

```python
import jax
import jax.numpy as jnp

import flax.nnx as nnx
import optax

from dataclasses import dataclass
from jax.sharding import PartitionSpec as P, reshard
import grain.python as pygrain
import pandas as pd
import tiktoken
import time
```

<!-- #region -->
## Define the miniGPT model with Flax and JAX automatic parallelism

### Leveraging JAX's data and tensor parallelism

One of the most powerful features of JAX is [device parallelism](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization) for SPMD.

- The data parallelism technique enables, for example, the training data to run via multiple parts (this is called sharding) - batches - in parallel and simultaneously across different devices, such as GPUs and Google TPUs. This allows to use larger batch sizes to speed up training.
- Tensor parallelism allows us to split the model parameter tensors across several devices (sharding model tensors).
- You can learn more about the basics of JAX parallelism in more detail in the [Introduction to parallel programming](https://jax.readthedocs.io/en/latest/sharded-computation.html) on the JAX documentation site.

In this example, we'll utilize a 4-way data parallel and 2-way tensor parallel setup.


### Making a mesh

To shard data in JAX, we must create a [`jax.sharding.Mesh`](https://jax.readthedocs.io/en/latest/jax.sharding.html#jax.sharding.Mesh). A *mesh* is a multidimensional NumPy array of JAX devices, where each axis of the mesh has a name, such as `'x'` or `'y'`. This will help encapsulate the information about the TPU resource organization for distributing computations across the devices. We'll make a mesh with two axes: `batch` for data parallelism and `model` for model parallelism.

To do this, we call `jax.make_mesh` with the size and name of each axis. The call below to `jax.set_mesh` returns a context manager, which we'd use if we wanted to only set the current mesh temporarily. As we'll use the same mesh for all the code in this notebook, however, we can ignore the context manager. 
<!-- #endregion -->

```python
_ = jax.set_mesh(jax.make_mesh((2, 1), ('batch', 'model')))
```

We will use the GPT-2 tokenizer from the [Tiktoken](https://github.com/openai/tiktoken) library:

```python
tokenizer = tiktoken.get_encoding("gpt2")
```

To leverage model parallelism, we need to instruct the JAX compiler how to shard the model tensors across the TPU devices. To do this, we initialize the model's variables with the metadata `out_sharding` set to a `PartitionSpec`. A `PartitionSpec` is just a wrapper around a tuple of names. The elements of this tuple should describe how an input dimension is partitioned across mesh dimensions. For example, if `out_sharding=P('x', 'y')` the first dimension of data will be sharded across `x` axis of the mesh, and the second one across the `y` axis.

```python
class TransformerBlock(nnx.Module):
    """ A single Transformer block.

    Each Transformer block processes input sequences via self-attention and feed-forward networks.

    Args:
        embed_dim (int): Embedding dimensionality.
        num_heads (int): Number of attention heads.
        ff_dim (int): Dimensionality of the feed-forward network.
        rngs (flax.nnx.Rngs): A Flax NNX stream of JAX PRNG keys.
        rate (float): Dropout rate. Defaults to 0.1.
    """
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, *, rngs: nnx.Rngs, rate: float = 0.1):
        self.mha = nnx.MultiHeadAttention(num_heads=num_heads,
              in_features=embed_dim,
              kernel_metadata={'out_sharding': P(None, 'model')},
              bias_metadata={'out_sharding': P('model')},
              rngs=rngs)
        self.dropout1 = nnx.Dropout(rate=rate, rngs=rngs)
        self.layer_norm1 = nnx.LayerNorm(epsilon=1e-6,
             num_features=embed_dim,
             scale_metadata={'out_sharding': P('model')},
             bias_metadata={'out_sharding': P('model')},
             rngs=rngs)
        self.mlp = nnx.Sequential(
            nnx.Linear(in_features=embed_dim,
              out_features=ff_dim,
              kernel_metadata={'out_sharding': P(None, 'model')},
              bias_metadata={'out_sharding': P('model')},
              rngs=rngs),
            nnx.relu, 
            nnx.Linear(in_features=ff_dim,
              out_features=embed_dim,
              kernel_metadata={'out_sharding': P(None, 'model')},
              bias_metadata={'out_sharding': P('model')},
              rngs=rngs),
            nnx.Dropout(rate=rate, rngs=rngs))
        self.layer_norm2 = nnx.LayerNorm(epsilon=1e-6,
         num_features=embed_dim,
         scale_metadata={'out_sharding': P('model')},
         bias_metadata={'out_sharding': P('model')},
         rngs=rngs)


    # Apply the Transformer block to the input sequence.
    def __call__(self, inputs):
        # Instantiate the causal attention mask.
        attention_output = self.mha(
            inputs_q=inputs,
            is_causal=True,
            decode=False
        )
        attention_output = self.dropout1(attention_output)
        out1 = self.layer_norm1(inputs + attention_output)
        ffn_output = self.mlp(out1)
        return self.layer_norm2(out1 + ffn_output)
```

```python
class TokenAndPositionEmbedding(nnx.Module):
    """ Combines token embeddings (words in an input sentence) with
    positional embeddings (the position of each word in a sentence).

    Args:
        maxlen (int): Matimum sequence length.
        vocal_size (int): Vocabulary size.
        embed_dim (int): Embedding dimensionality.
        rngs (flax.nnx.Rngs): A Flax NNX stream of JAX PRNG keys.
    """
    def __init__(self, maxlen: int, vocab_size: int, embed_dim: int, *, rngs: nnx.Rngs):
        # Initialize token embeddings (using `flax.nnx.Embed`).
        # Each unique word has an embedding vector.
        self.token_emb = nnx.Embed(num_embeddings=vocab_size, features=embed_dim, rngs=rngs)
        # Initialize positional embeddings (using `flax.nnx.Embed`).
        self.pos_emb = nnx.Embed(num_embeddings=maxlen, features=embed_dim, rngs=rngs)

    # Takes a token sequence (integers) and returns the combined token and positional embeddings.
    def __call__(self, x):
        # Generate a sequence of positions for the input tokens.
        positions = jnp.arange(0, x.shape[1])[None, :]
        # Look up the positional embeddings for each position in the input sequence.
        position_embedding = self.pos_emb(positions)
        # Look up the token embeddings for each token in the input sequence.
        token_embedding = self.token_emb(x, out_sharding=jax.typeof(x).sharding)
        # Combine token and positional embeddings.
        return token_embedding + position_embedding
```

```python
class MiniGPT(nnx.Module):
    """ A miniGPT transformer model, inherits from `flax.nnx.Module`.

    Args:
        maxlen (int): Maximum sequence length.
        vocab_size (int): Vocabulary size.
        embed_dim (int): Embedding dimensionality.
        num_heads (int): Number of attention heads.
        feed_forward_dim (int): Dimensionality of the feed-forward network.
        num_transformer_blocks (int): Number of transformer blocks. Each block contains attention and feed-forward networks.
        rngs (nnx.Rngs): A Flax NNX stream of JAX PRNG keys.
    """
    # Initialize miniGPT model components.
    def __init__(self, maxlen: int, vocab_size: int, embed_dim: int, num_heads: int, feed_forward_dim: int, num_transformer_blocks: int, rngs: nnx.Rngs):
        # Initiliaze the `TokenAndPositionEmbedding` that combines token and positional embeddings.
        self.embedding_layer = TokenAndPositionEmbedding(
                    maxlen, vocab_size, embed_dim, rngs=rngs
                )
        # Create a list of `TransformerBlock` instances.
        # Each block processes input sequences using attention and feed-forward networks.
        self.transformer_blocks = nnx.Sequential(*[TransformerBlock(
            embed_dim, num_heads, feed_forward_dim, rngs=rngs
        ) for _ in range(num_transformer_blocks)])
        # Initialize the output `flax.nnx.Linear` layer producing logits over the vocabulary for next-token prediction.
        self.output_layer = nnx.Linear(in_features=embed_dim,
                                       out_features=vocab_size,
                                       kernel_metadata={'out_sharding': P(None, 'model')},
                                       bias_metadata={'out_sharding': P('model')},
                                       rngs=rngs)

    def __call__(self, inputs):
        # Pass the input tokens through the `embedding_layer` to get token embeddings.
        x = self.embedding_layer(inputs)
        # Apply each transformer block sequentially to the embedded input
        x = self.transformer_blocks(x)
        # Pass the output of the transformer blocks through the output layer,
        # and obtain logits for each token in the vocabulary (for next token prediction).
        return reshard(self.output_layer(x), jax.typeof(inputs).sharding)

    def sample_from(self, logits):
        logits, indices = jax.lax.top_k(logits, k=top_k)
        logits = nnx.softmax(logits)
        return jax.random.choice(jax.random.PRNGKey(0), indices, p=logits)

    @nnx.jit(donate_argnums=(1,))
    def generate_step(self, padded_tokens, sample_index):
        logits = self(padded_tokens)
        next_token = self.sample_from(logits[0][sample_index])
        return next_token

    def generate_text(self, max_tokens, start_tokens):
        generated = []
        for i in range(max_tokens):
            sample_index = len(start_tokens) + len(generated) - 1

            padded_tokens = jnp.array((start_tokens + generated + [0] * (maxlen - len(start_tokens) - len(generated))))[None, :]
            next_token = int(self.generate_step(padded_tokens, sample_index))
            if next_token == tokenizer.encode('<|endoftext|>', allowed_special={'<|endoftext|>'})[0]:
              break
            generated.append(next_token)
        return tokenizer.decode(start_tokens + generated)

# Creates the miniGPT model with 4 transformer blocks.
def create_model(rngs):
    return MiniGPT(maxlen, vocab_size, embed_dim, num_heads, feed_forward_dim, num_transformer_blocks=4, rngs=rngs)
```

Set some hyperparameters.

```python
vocab_size = tokenizer.n_vocab
num_transformer_blocks = 8
maxlen = 256
embed_dim = 256
num_heads = 8
feed_forward_dim = 256
batch_size = 144
num_epochs = 1
top_k = 10
```

## Loading and preprocessing the data

Data loading and preprocessing with [Grain](https://github.com/google/grain).

```python
@dataclass
class TextDataset:
    data: list
    maxlen: int

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        # Use Tiktoken for tokenization
        encoding = tokenizer.encode(self.data[idx], allowed_special={'<|endoftext|>'})[:self.maxlen]  # Tokenize and truncate
        return encoding + [0] * (self.maxlen - len(encoding))  # Pad to maxlen

def load_and_preprocess_data(file_path, batch_size, maxlen):

    with open(file_path, 'r') as f:
      text = f.read()

    stories = text.split('<|endoftext|>')
    stories = [story+'<|endoftext|>' for story in stories if story.strip()]
    df = pd.DataFrame({'text': stories})
    data = df['text'].dropna().tolist()
    dataset = TextDataset(data, maxlen)

    sampler = pygrain.IndexSampler(
        len(dataset),
        shuffle=False,
        seed=42,
        shard_options=pygrain.NoSharding(),
        num_epochs=num_epochs,
    )

    dl = pygrain.DataLoader(
        data_source=dataset,
        sampler=sampler,
        operations=[pygrain.Batch(batch_size=batch_size, drop_remainder=True)],
    )

    return dl

text_dl = load_and_preprocess_data('TinyStories-train.txt', batch_size, maxlen)
```

## Defining the loss function and training step function

```python
# Defines the loss function using `optax.softmax_cross_entropy_with_integer_labels`.
def loss_fn(model, batch):
    logits = model(batch[0])
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=batch[1]).mean()
    return loss, logits

# Define the training step with the `flax.nnx.jit` transformation decorator.
@nnx.jit(donate_argnums=(0, 1, 3))
def train_step(model: MiniGPT, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits, lables=batch[1])
    optimizer.update(model, grads)
```

## Training the model

For data parallelism, we must shard the training data along the `batch` axis. To do this, we can use `jax.device_put`, which takes a `PartitionSpec` of how to shard its argument. We are also using the `jax.vmap` transformation to produce the target sequences faster.

```python
model = create_model(rngs=nnx.Rngs(0))
```

```python
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
metrics = nnx.MultiMetric(
    loss=nnx.metrics.Average("loss"),
)
rng = jax.random.PRNGKey(0)
```

```python
start_prompt = "Once upon a time"
start_tokens = tokenizer.encode(start_prompt)[:maxlen]
model.generate_text(maxlen, start_tokens)
```

```python outputId="5dd06dca-f030-4927-a9b6-35d412da535c"
metrics_history = {
    "train_loss": [],
}

prep_target_batch = jax.vmap(
    lambda tokens: jnp.concatenate((tokens[1:], jnp.array([0])))
)

step = 0
for epoch in range(num_epochs):
    start_time = time.time()
    for batch in text_dl:
        if len(batch) % len(jax.devices()) != 0:
            continue  # skip the remaining elements
        input_batch = jnp.stack(batch).T
        target_batch = prep_target_batch(input_batch)
        train_step(
            model,
            optimizer,
            metrics,
            jax.device_put(
                (input_batch, target_batch), P("batch", None)
            ),
        )

        if (step + 1) % 200 == 0:
            for metric, value in metrics.compute().items():
                metrics_history[f"train_{metric}"].append(value)
            metrics.reset()

            elapsed_time = time.time() - start_time
            print(
                f"\n\nStep {step + 1}, Loss: {metrics_history['train_loss'][-1]}, Elapsed Time: {elapsed_time:.2f} seconds"
            )
            start_time = time.time()

            print("Generated text:")
            print(model.generate_text(maxlen, start_tokens))

        step += 1

# Final text generation
print("Final generated text:")
generated_text = model.generate_text(maxlen, start_tokens)
```

Visualize the training loss.

```python outputId="7cafe711-1ae4-4eb9-fd37-e1bde54cbfc5"
import matplotlib.pyplot as plt
plt.plot(metrics_history['train_loss'])
plt.title('Training Loss')
plt.xlabel('Step % 200')
plt.ylabel('Loss')
plt.show()
```

As you can see, the model goes from generating completely random words at the beginning to generating sensible tiny stories at the end of the training. So essentially we have pretrained a small LLM to write tiny stories for us.


## Saving the checkpoint

Save the model checkpoint.

```python outputId="3467b8ba-ce05-42f0-fb89-75922cc91e31"
import orbax.checkpoint as orbax
from pathlib import Path

state = nnx.state(model)
checkpoint_path = Path('checkpoint').resolve()
checkpointer = orbax.PyTreeCheckpointer()
checkpointer.save(checkpoint_path, args=orbax.args.PyTreeSave(state), force=True)
```

## Profiling for hyperparameter tuning


Load the tensorboard colab extension.

```python
%load_ext tensorboard
```

As we're going to be running this model a number of times, we need some scaffolding to more easily compare our work. For a baseline, we'll need to perform some warmup to guarantee that our code is JIT'd and that our TPUs are warm. For improved comparability, we'll only start tracing after we've finished warmup.

```python
trace_dir = "/tmp/jax-trace/"

def loop_step(batch, step):
    input_batch = jnp.stack(batch).T
    target_batch = prep_target_batch(input_batch)
    train_step(model, optimizer, metrics, jax.device_put((input_batch, target_batch), P('batch', None)))

def generate_trace():
    tracing_steps = 30
    warmup_steps = 5
    for current_step in range(warmup_steps + tracing_steps):
        if current_step == warmup_steps:
            jax.profiler.start_trace(trace_dir)
        with jax.profiler.StepTraceAnnotation("train", step_num=current_step):
            batch = next(text_dl)
            loop_step(batch, current_step)

    jax.profiler.stop_trace()
```

Now we'll perform some traces to compare results of different batch sizes. This will take several minutes as we need to reprocess our input data to prepare new batches each time.

```python
trace_dir = "/tmp/jax-trace-batch-comparison/"

batch_size = 64
text_dl = iter(load_and_preprocess_data('TinyStories-train.txt', batch_size, maxlen))
generate_trace()

batch_size = 256
text_dl = iter(load_and_preprocess_data('TinyStories-train.txt', batch_size, maxlen))
generate_trace()
```

Run Tensorboard with the Profiler Plugin to compare our runs. Runs are listed in order from newest to oldest, so the top run in the list will be have `batch_size = 256`.

The key metrics to focus on here for this hyperparameter are Framework Op Placement and Average Step Time.

In general, we want to maximize the Framework Op Placement on the device while minimizing the step time per training example. In this case, we can see that increasing the batch size from 64 -> 256 achieves both of those. FLOPS increases from 16% to 27%. Average Step Time increase from 100ms to 260ms, however we increased our batch size by 300%. This means we move from 1.5ms per training example to 1.02ms per training example.

```python
%tensorboard --logdir $trace_dir --port 6006
```

Next, we can explore alternative parallelism methods. Previously, we used 4-way data parallelism and 2-way model parallelism. 8-way data parallelism is another popular way of distributing work. Let's compare results between them. To switch to 8-way data parallel, we'll replace mesh with:

`jax.make_mesh((8, 1), ('batch', 'model'))`

JAX will automatically figure out how to shard the model and data to use the new partition strategy and nothing else need to be done. Re-connect the TPU runtime and run it again to see how it runs.

How simple and powerful is this! And that's the beauty of JAX automatic parallelism.

```python
trace_dir = "/tmp/jax-trace-parallelism-comparison/"

mesh = Mesh(mesh_utils.create_device_mesh((4, 2)), ('batch', 'model'))
generate_trace()

mesh = Mesh(mesh_utils.create_device_mesh((8, 1)), ('batch', 'model'))
generate_trace()
```

Once again we'll run tensorboard.

Looking at the results, we see that the step times are nearly the same, however the FLOPS Utilization is at 13% for 8-way data parallelism compared to 27% or 4-way data parallelism.

By looking at the Trace Viewer tool and looking under each TPU's ops, we can see that the TPUs spend a large amount of time idle while waiting for the host, as well as spending a good amount of time in `reduce_sum` operations.

```python
%tensorboard --logdir=$trace_dir
```

By changing hyperparameters and comparing profiles, we're able to gain significant insights into our bottlenecks and limitations. These are just two examples of hyperparameters to tune, but plenty more of them will have significant effects on training speed and resource utilization.
