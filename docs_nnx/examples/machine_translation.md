---
jupyter:
  jupytext:
    formats: ipynb,md
    main_language: python
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
---

# Machine Translation with encoder-decoder transformer model


This tutorial is adapted from [Keras' documentation on English-to-Spanish translation with a sequence-to-sequence Transformer](https://keras.io/examples/nlp/neural_machine_translation_with_transformer/), which is itself an adaptation from the book [Deep Learning with Python, Second Edition by François Chollet](https://www.manning.com/books/deep-learning-with-python-second-edition)

We step through an encoder-decoder transformer in JAX and train a model for English->Spanish translation.

```python
# !pip install tiktoken grain flax optax
```

```python
import pathlib
import random
import string
import re
import numpy as np

import jax.numpy as jnp
import optax

from flax import nnx

import tiktoken
import grain.python as grain
import tqdm
```

## Pull down data to temp and extract into memory

There are lots of ways to get this done, but for simplicity and clear visibility into what's happening this is downloaded to a temporary directory, extracted there, and read into a python object with processing.

```python
import requests
import zipfile
import tempfile

url = "http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip"

with tempfile.TemporaryDirectory() as temp_dir:
    temp_path = pathlib.Path(temp_dir)
    zip_file_path = temp_path / "spa-eng.zip"

    response = requests.get(url)
    zip_file_path.write_bytes(response.content)

    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(temp_path)

    text_file = temp_path / "spa-eng" / "spa.txt"

    with open(text_file) as f:
        lines = f.read().split("\n")[:-1]
    text_pairs = []
    for line in lines:
        eng, spa = line.split("\t")
        spa = "[start] " + spa + " [end]"
        text_pairs.append((eng, spa))
```

## Build train/validate/test pair sets
We'll stay close to the original tutorial so it's clear how to follow what's the same vs what's different; one early difference is the choice to go with an off-the-shelf encoder/tokenizer in tiktoken. Specifically "cl100k_base" - it has a wide range of language understanding and it's fast.

```python outputId="cf7fc8d5-029d-48d6-d739-95c53bdc9b38"
random.shuffle(text_pairs)
num_val_samples = int(0.15 * len(text_pairs))
num_train_samples = len(text_pairs) - 2 * num_val_samples
train_pairs = text_pairs[:num_train_samples]
val_pairs = text_pairs[num_train_samples : num_train_samples + num_val_samples]
test_pairs = text_pairs[num_train_samples + num_val_samples :]

print(f"{len(text_pairs)} total pairs")
print(f"{len(train_pairs)} training pairs")
print(f"{len(val_pairs)} validation pairs")
print(f"{len(test_pairs)} test pairs")
```

```python
tokenizer = tiktoken.get_encoding("cl100k_base")
```

We strip out punctuation to keep things simple and in line with the original tutorial - the `[` `]` are kept in so that our `[start]` and `[end]` formatting is preserved.

```python
strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")

vocab_size = tokenizer.n_vocab
sequence_length = 20
```

```python
def custom_standardization(input_string):
    lowercase = input_string.lower()
    return re.sub(f"[{re.escape(strip_chars)}]", "", lowercase)
```

```python
def tokenize_and_pad(text, tokenizer, max_length):
    tokens = tokenizer.encode(text)[:max_length]
    padded = tokens + [0] * (max_length - len(tokens)) if len(tokens) < max_length else tokens ##assumes list-like - (https://github.com/openai/tiktoken/blob/main/tiktoken/core.py#L81 current tiktoken out)
    return padded
```

```python
def format_dataset(eng, spa, tokenizer, sequence_length):
    eng = custom_standardization(eng)
    spa = custom_standardization(spa)
    eng = tokenize_and_pad(eng, tokenizer, sequence_length)
    spa = tokenize_and_pad(spa, tokenizer, sequence_length)
    return {
            "encoder_inputs": eng,
            "decoder_inputs": spa[:-1],
            "target_output": spa[1:],
            }
```

```python
train_data = [format_dataset(eng, spa, tokenizer, sequence_length) for eng, spa in train_pairs]
val_data = [format_dataset(eng, spa, tokenizer, sequence_length) for eng, spa in val_pairs]
test_data = [format_dataset(eng, spa, tokenizer, sequence_length) for eng, spa in test_pairs]
```

At this point we've extracted the data, applied formatting, and tokenized the phrases with padding. The data is kept in train/validate/test sets that each have dictionary entries, which look like the following:

```python outputId="017a7188-ff72-4f53-b725-be16f48cdd9f"
## data selection example
print(train_data[135])
```

The output should look something like

{'encoder_inputs': [9514, 265, 3339, 264, 2466, 16930, 1618, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'decoder_inputs': [29563, 60, 1826, 7206, 71086, 37116, 653, 16109, 1493, 54189, 510, 408, 60, 0, 0, 0, 0, 0, 0], 'target_output': [60, 1826, 7206, 71086, 37116, 653, 16109, 1493, 54189, 510, 408, 60, 0, 0, 0, 0, 0, 0, 0]}


## Define Transformer components: Encoder, Decoder, Positional Embed

In many ways this is very similar to the original source, with `ops` changing to `jnp` and `keras` or `layers` becoming `nnx`. Certain module-specific arguments come and go, like the rngs attached to most things in the updated version, and decode=False in the MultiHeadAttention call.

```python
class TransformerEncoder(nnx.Module):
    def __init__(self, embed_dim: int, dense_dim: int, num_heads: int, rngs: nnx.Rngs, **kwargs):
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads

        self.attention = nnx.MultiHeadAttention(num_heads=num_heads,
                                          in_features=embed_dim,
                                          decode=False,
                                          rngs=rngs)
        self.dense_proj = nnx.Sequential(
                nnx.Linear(embed_dim, dense_dim, rngs=rngs),
                nnx.relu,
                nnx.Linear(dense_dim, embed_dim, rngs=rngs),
        )

        self.layernorm_1 = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.layernorm_2 = nnx.LayerNorm(embed_dim, rngs=rngs)

    def __call__(self, inputs):
        attention_output = self.attention(
            inputs_q = inputs, inputs_k = inputs, inputs_v = inputs, decode = False
        )
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)


class PositionalEmbedding(nnx.Module):
    def __init__(self, sequence_length: int, vocab_size: int, embed_dim: int, rngs: nnx.Rngs, **kwargs):
        self.token_embeddings = nnx.Embed(num_embeddings=vocab_size, features=embed_dim, rngs=rngs)
        self.position_embeddings = nnx.Embed(num_embeddings=sequence_length, features=embed_dim, rngs=rngs)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def __call__(self, inputs):
        length = inputs.shape[1]
        positions = jnp.arange(0, length)[None, :]
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        else:
            return jnp.not_equal(inputs, 0)

class TransformerDecoder(nnx.Module):
    def __init__(self, embed_dim: int, latent_dim: int, num_heads: int, rngs: nnx.Rngs, **kwargs):
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = nnx.MultiHeadAttention(num_heads=num_heads,
                                  in_features=embed_dim,
                                  decode=False,
                                  rngs=rngs)
        self.attention_2 = nnx.MultiHeadAttention(num_heads=num_heads,
                                  in_features=embed_dim,
                                  decode=False,
                                  rngs=rngs)

        self.dense_proj = nnx.Sequential(
                nnx.Linear(embed_dim, latent_dim, rngs=rngs),
                nnx.relu,
                nnx.Linear(latent_dim, embed_dim, rngs=rngs),
        )
        self.layernorm_1 = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.layernorm_2 = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.layernorm_3 = nnx.LayerNorm(embed_dim, rngs=rngs)

    def __call__(self, inputs, encoder_outputs):
        causal_mask = nnx.make_causal_mask(inputs[:,:,0])
        attention_output_1 = self.attention_1(
            inputs_q=inputs, inputs_v=inputs, inputs_k=inputs,  mask=causal_mask
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            inputs_q=out_1,
            inputs_v=encoder_outputs,
            inputs_k=encoder_outputs
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)
```

Here we finally use our earlier encoder, decoder, and positional embed classes to construct the Model that we'll train and later use for inference.

```python
class TransformerModel(nnx.Module):
    def __init__(self, sequence_length: int, vocab_size: int, embed_dim: int, latent_dim: int, num_heads: int, dropout_rate: float, rngs: nnx.Rngs):
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.encoder = TransformerEncoder(embed_dim, latent_dim, num_heads, rngs=rngs)
        self.positional_embedding = PositionalEmbedding(sequence_length, vocab_size, embed_dim, rngs=rngs)
        self.decoder = TransformerDecoder(embed_dim, latent_dim, num_heads, rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.dense = nnx.Linear(embed_dim, vocab_size, rngs=rngs)

    def __call__(self, encoder_inputs: jnp.array, decoder_inputs: jnp.array):
        x = self.positional_embedding(encoder_inputs)
        encoder_outputs = self.encoder(x)

        x = self.positional_embedding(decoder_inputs)
        decoder_outputs = self.decoder(x, encoder_outputs)
        decoder_outputs = self.dropout(decoder_outputs, deterministic=False)

        logits = self.dense(decoder_outputs)
        return logits
```

## Build out Data Loader and Training Definitions
It can be more computationally efficient to use pygrain for the data load stage, but this way it's abundandtly clear what's happening: data pairs go in and sets of jnp arrays come out, in step with our original dictionaries. 'Encoder_inputs', 'decoder_inputs' and 'target_output'.

```python
batch_size = 512 #set here for the loader and model train later on

class CustomPreprocessing(grain.MapTransform):
    def __init__(self):
        pass

    def map(self, data):
        return {
            "encoder_inputs": np.array(data["encoder_inputs"]),
            "decoder_inputs": np.array(data["decoder_inputs"]),
            "target_output": np.array(data["target_output"]),
        }

train_sampler = grain.IndexSampler(
    len(train_data),
    shuffle=True,
    seed=12,                        # Seed for reproducibility
    shard_options=grain.NoSharding(), # No sharding since it's a single-device setup
    num_epochs=1,                    # Iterate over the dataset for one epoch
)

val_sampler = grain.IndexSampler(
    len(val_data),
    shuffle=False,
    seed=12,
    shard_options=grain.NoSharding(),
    num_epochs=1,
)

train_loader = grain.DataLoader(
    data_source=train_data,
    sampler=train_sampler,                 # Sampler to determine how to access the data
    worker_count=4,                        # Number of child processes launched to parallelize the transformations
    worker_buffer_size=2,                  # Count of output batches to produce in advance per worker
    operations=[
        CustomPreprocessing(),
        grain.Batch(batch_size=batch_size, drop_remainder=True),
    ]
)

val_loader = grain.DataLoader(
    data_source=val_data,
    sampler=val_sampler,
    worker_count=4,
    worker_buffer_size=2,
    operations=[
        CustomPreprocessing(),
        grain.Batch(batch_size=batch_size),
    ]
)
```

Optax doesn't have the identical loss function that the source tutorial uses, but this softmax cross entropy works well here - you can one_hot_encode if you don't use the `_with_integer_labels` version of the loss.

```python
def compute_loss(logits, labels):
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels)
    return jnp.mean(loss)
```

While in the original tutorial most of the model and training details happen inside keras, we make them explicit here in our step functions, which are later used in `train_one_epoch` and `eval_model`.

```python
@nnx.jit
def train_step(model, optimizer, batch):
    def loss_fn(model, train_encoder_input, train_decoder_input, train_target_input):
        logits = model(train_encoder_input, train_decoder_input)
        loss = compute_loss(logits, train_target_input)
        return loss

    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model, jnp.array(batch["encoder_inputs"]), jnp.array(batch["decoder_inputs"]), jnp.array(batch["target_output"]))
    optimizer.update(grads)
    return loss

@nnx.jit
def eval_step(model, batch, eval_metrics):
    logits = model(jnp.array(batch["encoder_inputs"]), jnp.array(batch["decoder_inputs"]))
    loss = compute_loss(logits, jnp.array(batch["target_output"]))
    labels = jnp.array(batch["target_output"])

    eval_metrics.update(
        loss=loss,
        logits=logits,
        labels=labels,
    )
```

Here, `nnx.MultiMetric` helps us keep track of general training statistics, while we make our own dictionaries to hold historical values

```python
eval_metrics = nnx.MultiMetric(
    loss=nnx.metrics.Average('loss'),
    accuracy=nnx.metrics.Accuracy(),
)

train_metrics_history = {
    "train_loss": [],
}

eval_metrics_history = {
    "test_loss": [],
    "test_accuracy": [],
}
```

```python
## Hyperparameters
rng = nnx.Rngs(0)
embed_dim = 256
latent_dim = 2048
num_heads = 8
dropout_rate = 0.5
vocab_size = tokenizer.n_vocab
sequence_length = 20
learning_rate = 1.5e-3
num_epochs = 10
```

```python
bar_format = "{desc}[{n_fmt}/{total_fmt}]{postfix} [{elapsed}<{remaining}]"
train_total_steps = len(train_data) // batch_size

def train_one_epoch(epoch):
    with tqdm.tqdm(
        desc=f"[train] epoch: {epoch}/{num_epochs}, ",
        total=train_total_steps,
        bar_format=bar_format,
        leave=True,
    ) as pbar:
        for batch in train_loader:
            loss = train_step(train_model, optimizer, batch)
            train_metrics_history["train_loss"].append(loss.item())
            pbar.set_postfix({"loss": loss.item()})
            pbar.update(1)


def evaluate_model(epoch):
    # Compute the metrics on the train and val sets after each training epoch.


    eval_metrics.reset()  # Reset the eval metrics
    for val_batch in val_loader:
        eval_step(model, val_batch, eval_metrics)

    for metric, value in eval_metrics.compute().items():
        eval_metrics_history[f'test_{metric}'].append(value)

    print(f"[test] epoch: {epoch + 1}/{num_epochs}")
    print(f"- total loss: {eval_metrics_history['test_loss'][-1]:0.4f}")
    print(f"- Accuracy: {eval_metrics_history['test_accuracy'][-1]:0.4f}")
```

```python
model = TransformerModel(sequence_length, vocab_size, embed_dim, latent_dim, num_heads, dropout_rate, rngs=rng)
train_model = nnx.view(model, deterministic=False, use_running_average=False)
eval_model = nnx.view(model, deterministic=True, use_running_average=True)

optimizer = nnx.ModelAndOptimizer(model, optax.adamw(learning_rate))
```

## Start the Training!
With our data loaders set and the model, optimizer, and epoch train/eval functions set up - time to finally press go - on a 3090, this is roughly 19GB VRAM and each epoch is roughly 18 seconds with batch_size set to 512.

```python outputId="a4f97db4-da4e-481b-f5c2-7137216e6380"
for epoch in range(num_epochs):
    train_one_epoch(epoch)
    evaluate_model(epoch)
```

We can then plot the loss over training time. That log-plot comes in handy here, or it's hard to appreciate the progress after 1000 steps or so.

```python outputId="c1cdf060-335f-4321-8982-7ed07d15db06"
import matplotlib.pyplot as plt

plt.plot(train_metrics_history["train_loss"], label="Loss value during the training")
plt.yscale('log')
plt.legend();
```

And eval set Loss and Accuracy - Accuracy does continue to rise, though it's hard-earned progress after about the 5th epoch. Based on the training statistics, it's fair to say the process starts overfitting after roughly that 5th epoch.

```python outputId="54680efe-bcf5-44df-e468-3054fb1e56bc"
fig, axs = plt.subplots(1, 2)
axs[0].set_title("Log loss value on eval set")
axs[0].plot(np.log(eval_metrics_history["test_loss"]))
axs[1].set_title("Accuracy on eval set")
axs[1].plot(eval_metrics_history["test_accuracy"])
plt.tight_layout();
```

## Use Model for Inference
After all that, the product of what we were working for: a trained model we can save and load for inference. For people using LLMs recently, this pattern may look rather familiar: an input sentence tokenized into an array and computed 'next' token-by-token. While many recent LLMs are decoder-only, this was an encoder/decoder architecture with the very specific english-to-spanish pattern baked in.

We've changed a couple things from the source 'use' function, here - because of the tokenizer used, things like `[start]` and `[end]` are no longer single tokens - instead `[start]` is `[29563, 60] = "[start" + "]"` and `[end]` is `[58308, 60] = "[end" + "]"` - thus we start with only a single token `[start` and can't only test on `last_token = "[end"]`. Otherwise, the main change here is that the input is assumed a single sentence, rather than batch inference.

```python
def decode_sequence(input_sentence):

    input_sentence = custom_standardization(input_sentence)
    tokenized_input_sentence = tokenize_and_pad(input_sentence, tokenizer, sequence_length)

    decoded_sentence = "[start"
    for i in range(sequence_length):
        tokenized_target_sentence = tokenize_and_pad(decoded_sentence, tokenizer, sequence_length)[:-1]
        predictions = eval_model(jnp.array([tokenized_input_sentence]), jnp.array([tokenized_target_sentence]))

        sampled_token_index = np.argmax(predictions[0,i, :]).item(0)
        sampled_token = tokenizer.decode([sampled_token_index])
        decoded_sentence += "" + sampled_token

        if decoded_sentence[-5:] == "[end]":
            break
    return decoded_sentence
```

```python
test_eng_texts = [pair[0] for pair in test_pairs]
```

```python
test_result_pairs = []
for _ in range(10):
    input_sentence = random.choice(test_eng_texts)
    translated = decode_sequence(input_sentence)

    test_result_pairs.append(f"[Input]: {input_sentence} [Translation]: {translated}")
```

## Test Results
For the model and the data, not too shabby - It's definitely spanish-ish. Though when 'making' friends, please don't confuse 'hacer' (to make) with 'comer' (to eat).

```python outputId="c35060a1-a0c6-43ad-e21c-b4dbff8e2536"
for i in test_result_pairs:
    print(i)
```

Example output from the above cell:

      [Input]: We're going to have a baby. [Translation]: [start] nosotros vamos a tener un bebé [end]
      [Input]: You drive too fast. [Translation]: [start] conducís demasiado rápido [end]
      [Input]: Let me know if there's anything I can do. [Translation]: [start] déjame saber si hay cualquier cosa que yo pueda hacer [end]
      [Input]: Let's go to the kitchen. [Translation]: [start] vayamos a la cocina [end]
      [Input]: Tom gasped. [Translation]: [start] tom se quedó sin aliento [end]
      [Input]: I was just hanging out with some of my friends. [Translation]: [start] estaba escquieto con algunos de mi amigos [end]
      [Input]: Tom is in the bathroom. [Translation]: [start] tom está en el cuarto de baño [end]
      [Input]: I feel safe here. [Translation]: [start] me siento segura [end]
      [Input]: I'm going to need you later. [Translation]: [start] me voy a necesitar después [end]
      [Input]: A party is a good place to make friends with other people. [Translation]: [start] una fiesta es un buen lugar de comer amigos con otras personas [end]
