- Start Date: 2021-08-23
- FLIP PR: [#1504](https://github.com/google/flax/pull/1504)
- FLIP Issue: [#1503](https://github.com/google/flax/issues/1503)


# Summary

This FLIP proposes to add a new modular trainining and evaluation API.


# Motivation

A unified training and evaluation API is crucial to the library adaptation.

Built-in support for training, evaluation, logging, checkpoints, and other instruments allow efficient development in Flax and lowering an entry barrier for new users, accelerating the library adaptation.

The proposed modular API defines a checkpoint format that may be utilized by various replaceable extensions.

A few built-in extensions aim to cover the most common use cases, while the lightweight core API intended to supply an interface for building custom extensions in a unified way. 


# Implementation

> The draft implementation may be found [here](https://github.com/manifest/flax-extra).


### Training loop

The core element of the API is a training loop. It serves the only purpose to yield a checkpoint at specified steps of the training.

```python
for checkpoint in train_loop:
    pass
```

The [checkpoint](https://github.com/manifest/flax-extra/blob/v0.0.1/src/flax_extra/checkpoint/_checkpoint.py#L9-L35) captures the training state at the particular step.

```python
class Checkpoint:
    ## Model variables.
    model_params: FrozenDict
    model_state: FrozenDict
    ## Optimizer state.
    optimizer_state: optax.OptState
    ## Training statistics.
    grads: FrozenDict
    loss: float
    n_completed_steps: int
    elapsed_time: float
    step: int
```

To create a [training loop](https://github.com/manifest/flax-extra/blob/v0.0.1/src/flax_extra/training/_loop.py#L155), one specifies:
- how to initialize a model (an `init` function of a linen module).
- how to propagate through the model (an `apply` function of a linen module).
- what training task to apply (e.g. an optimizer, a loss, and the training data stream).
- how often to yield a checkpoint.
- how many training  steps to perform in total.

```python
train_loop = TrainLoop(
    init=model().init,
    task=TrainTask(
        apply=model().apply,
        optimizer=optax.sgd(learning_rate=0.1),
        loss=categorical_cross_entropy,
        data=train_datastream,
    ),
    n_steps_per_checkpoint=25,
    n_steps=100,
)
```

In some situations, it may be useful to perform a single step.

```python
checkpoint = train_loop.next_step()
```

Or a number of steps remaining for the next checkpoint.

```python
checkpoint = train_loop.next_checkpoint()
```

An arbitrary number of steps may also be specified explicitly.

```python
for checkpoint in train_loop(n_steps=200):
    pass
```


### Evaluation loop

The evaluation loop performs a few steps evaluating a model (at the particular state described by a checkpoint obtained from the training loop) then returns the original checkpoint enhanced with averaged metrics.

The integration to the training loop is straight forward.

```python
for checkpoint in train_loop:
    summary = eval_loop(checkpoint)
```

The [summary](https://github.com/manifest/flax-extra/blob/v0.0.1/src/flax_extra/checkpoint/_summary.py#L12-L16) is just a checkpoint with associated metrics. Metrics are grouped by arbitrary label (such as "train" or "eval") to separate between different stages, data streams, etc.


```python
Metrics = Mapping[str, float]

class Summary(Checkpoint):
    metrics: Mapping[str, Metrics]
```

To create an [evaluation loop](https://github.com/manifest/flax-extra/blob/v0.0.1/src/flax_extra/evaluation/_loop.py#L101), one specifies:
- how to propagate through the model (an `apply` function of a linen module).
- metric functions to use in the evaluation, with corresponding labels.
- how many evaluation steps to perform.

```python
EvalLoop(
    task=EvalTask(
        apply=model().apply,
        metrics=dict(lnpp=categorical_cross_entropy),
        data=eval_datastream,
    ),
    n_steps=10,
)
```


# Extensions

The [Checkpoint](https://github.com/manifest/flax-extra/blob/v0.0.1/src/flax_extra/checkpoint/_checkpoint.py#L9-L35) and [Summary](https://github.com/manifest/flax-extra/blob/v0.0.1/src/flax_extra/checkpoint/_summary.py#L12-L16) dataclasses capture information that may be downstreamed to various replaceable extensions. The following sections cover extensions for the most common use cases.


### Logging extension

The [SummaryLogger](https://github.com/manifest/flax-extra/blob/v0.0.1/src/flax_extra/checkpoint/_summary_logger.py#L9) provides human-readable reports to stdout. These reports include training statistics along with evaluation metrics for observed checkpoints.

```python
logger = SummaryLogger()

for checkpoint in train_loop:
    summary = eval_loop(checkpoint)
    _ = logger(summary)
```

```
A checkpoint was loaded in 0.02 seconds.
Total model initialization time is 9.53 seconds.
The lowest value for the metric eval/lnpp is set to 5.54797554.
Total number of trainable weights: 657920 ~ 2.5 MB.

Step      7: Ran 1 train steps in 15.91 seconds
Step      7: train seconds_per_step | 15.90994287
Step      7: train gradients_l2norm | 0.00417352
Step      7: train   weights_l2norm | 19.82766914
Step      7: train             loss | 7.84606361
Step      7: eval              lnpp | 5.54795837
```


### Tensorboard extension

The [SummaryWriter](https://github.com/manifest/flax-extra/blob/v0.0.1/src/flax_extra/checkpoint/_summary_writer.py#L10) writes summaries to the local file system in Tensorboard file format.

```python
summary_writer = SummaryWriter(output_dir="/tmp/tensorboard")

for checkpoint in train_loop:
    summary = eval_loop(checkpoint)
    _ = summary_writer(summary)
```

To access summaries in Tensorboard, one simply runs.

```bash
tensorboard --logdir "/tmp/tensorboard"
```


### Regular checkpoint extension

The [CheckpointFileWriter](https://github.com/manifest/flax-extra/blob/v0.0.1/src/flax_extra/checkpoint/_checkpoint_file_writer.py#L13) stores each observed checkpoint on the local file system.

```python
checkpoint_writer = CheckpointFileWriter(output_dir="/tmp/checkpoints")

for checkpoint in train_loop:
    _ = checkpoint_writer(checkpoint)
```

To load the latest checkpoint from the local file system into the training loop, [CheckpointFileReader](https://github.com/manifest/flax-extra/blob/v0.0.1/src/flax_extra/checkpoint/_checkpoint_file_reader.py#L11) may be used.


```python
train_loop = TrainLoop(
    init=CheckpointFileReader(
        dir="/tmp/checkpoints",
        target=model().init,
    ),
    ...
)
```


### The best metric checkpoint extension

The [LowestCheckpointFileWriter](https://github.com/manifest/flax-extra/blob/v0.0.1/src/flax_extra/checkpoint/_best_checkpoint_file_writer.py#L92) and [HighestCheckpointFileWriter](https://github.com/manifest/flax-extra/blob/v0.0.1/src/flax_extra/checkpoint/_best_checkpoint_file_writer.py#L126) make it possible to store only checkpoints with the best observed metric value.

For example, the following writer stores a checkpoint if only the lowest log perplexity (lnpp) metric value was observed.

```python
checkpoint_writer = LowestCheckpointFileWriter(
    output_dir="/tmp/checkpoints",
    metric="lnpp",
)

for checkpoint in train_loop:
    _ = checkpoint_writer(checkpoint)
```

Similarly, the [LowestCheckpointFileReader](https://github.com/manifest/flax-extra/blob/v0.0.1/src/flax_extra/checkpoint/_best_checkpoint_file_reader.py#L13) and [HigestCheckpointFileReader](https://github.com/manifest/flax-extra/blob/v0.0.1/src/flax_extra/checkpoint/_best_checkpoint_file_reader.py#L44) can be used to initialize the training loop.

```python
train_loop = TrainLoop(
    init=LowestCheckpointFileReader(
        dir="/tmp/checkpoints",
        target=model().init,
        metric="lnpp",
    ),
    ...
)
```


# Complete example

> A notebook with the code may be found [here](https://flax-extra.readthedocs.io/en/latest/notebooks/training).

An example of training pipeline with extensions described above.

```python
import jax
from jax import numpy as jnp
import optax
from redex import combinator as cb
from flax_extra import random
from flax_extra.training import TrainLoop, TrainTask
from flax_extra.evaluation import EvalLoop, EvalTask
from flax_extra.checkpoint import (
    SummaryLogger,
    SummaryWriter,
    CheckpointFileReader,
    CheckpointFileWriter,
    LowestCheckpointFileWriter,
)
from flax_extra.model import RNNLM

MAX_LENGTH = 256
BATCH_SIZE = 32
VOCAB_SIZE = 2 ** 8
D_MODEL = 128

model = RNNLM

collections = dict(
    init=["params","carry","dropout"],
    apply=["params","carry","dropout"],
)
config = dict(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    n_layers=2,
)

def presudo_data_steam(shape, bounds, rnkey):
    minval, maxval = bounds
    while True:
        x = jax.random.uniform(
            key=rnkey,
            shape=shape,
            minval=minval,
            maxval=maxval,
        ).astype(jnp.int32)
        yield x, x

def categorical_cross_entropy(outputs, targets):
    n_categories = outputs.shape[-1]
    loss = optax.softmax_cross_entropy(
        outputs,
        jax.nn.one_hot(targets, n_categories),
    )
    return jnp.mean(loss)

rnkeyg = random.sequence(seed=0)
train_datag = presudo_data_steam(shape=(BATCH_SIZE,MAX_LENGTH), bounds=(1,VOCAB_SIZE), rnkey=next(rnkeyg))
eval_datag = presudo_data_steam(shape=(BATCH_SIZE,MAX_LENGTH), bounds=(1,VOCAB_SIZE), rnkey=next(rnkeyg))

train_loop = TrainLoop(
    init=CheckpointFileReader(dir="/tmp/checkpoints", target=model(**config).init),
    task=TrainTask(
        apply=model(**config | dict(deterministic=False)).apply,
        optimizer=optax.sgd(learning_rate=0.1, momentum=0.9),
        loss=categorical_cross_entropy,
        data=train_datag,
    ),
    collections=collections,
    mutable_collections=True,
    n_steps_per_checkpoint=3,
    rnkey=next(rnkeyg),
    n_steps=8,
)

process_checkpoint = cb.serial(
    EvalLoop(
        task=EvalTask(
            apply=model(**config).apply,
            metrics=dict(lnpp=categorical_cross_entropy),
            data=eval_datag,
        ),
        collections=collections,
        rnkey=next(rnkeyg),
        n_steps=2,
    ),
    SummaryLogger(),
    SummaryWriter(output_dir="/tmp/tensorboard"),
    CheckpointFileWriter(output_dir="/tmp/checkpoints"),
    LowestCheckpointFileWriter(output_dir="/tmp/checkpoints", metric="lnpp"),
)

for checkpoint in train_loop:
    _ = process_checkpoint(checkpoint)
```
