# Refactor RNNCellBase in FLIP

Authors: Cristian Garcia, Marcus Chiam, Jasmijn Bastings

 - Start Date: May 1, 2023
 - FLIP Issue: [TBD]
 - FLIP PR: #3053
 - Status: Implemented

## Summary
This proposal aims to improve the usability of the `RNNCellBase` class by refactoring the `initialize_carry` method and other relevant components.

## Motivation

Currently, `initialize_carry` is used to both initialize the carry and pass crucial metadata like the number of features. The API can be unintuitive as it requires users to manually calculate things that could typically be inferred by the modules themselves, such as the shape of batch dimensions and the shape of feature dimensions.

### Example: ConvLSTM
The current API can be unintuitive in cases like `ConvLSTM` where a the `size` parameter contains both the input image shape and output feature dimensions:

```python
x = jnp.ones((2, 4, 4, 3)) # (batch, *image_shape, channels)

#                                        image shape: vvvvvvv
carry = nn.ConvLSTMCell.initialize_carry(key1, (16,), (64, 64, 16))
#                                   batch size: ^^             ^^ :output features

lstm = nn.ConvLSTMCell(features=6, kernel_size=(3, 3))
(carry, y), initial_params = lstm.init_with_output(key2, carry, x)
```

This FLIP will propose some changes to `initialize_carry` such that the previous example can be simplified to:

```python
x = jnp.ones((2, 4, 4, 3)) # (batch, *image_shape, channels)

lstm = nn.ConvLSTMCell(features=6, kernel_size=(3, 3))
carry = lstm.initialize_carry(key1, input_shape=x.shape)

(carry, y), initial_params = lstm.init_with_output(key2, carry, x)
```

## Implementation
The proposal suggests the following changes:

### initialize_carry
`initialize_carry` should be refactored as an instance method with the following signature:

```python
def initialize_carry(self, key, sample_input):
```

`sample_input` should be an array of the same shape that will be processed by the cell, excluding the time axis.

### Refactor RNNCellBase subclasses
`RNNCellBase` should be refactored to include the metadata required to initialize the cell and execute its forward pass. For `LSTMCell` and `GRUCell`, this means adding a `features` attribute that should be provided by the user upon construction. This change aligns with the structure of most other `Module`s, making them more familiar to users.

```python
x = jnp.ones((2, 100, 10)) # (batch, time, features)

cell = nn.LSTMCell(features=32)
carry = cell.initialize_carry(PRNGKey(0), x[:, 0]) # sample input

(carry, y), variables = cell.init_with_output(PRNGKey(1), carry, x)
```

### num_feature_dims
To simplify the handling of `RNNCellBase` instances in abstractions like `RNN`, each cell should implement the `num_feature_dims` property. For most cells, such as `LSTMCell` and `GRUCell`, this is always 1. For cells like `ConvLSTM`, this depends on their `kernel_size`.

## Discussion
### Alternative Approaches
* To eliminate the need for `num_feature_dims`, `RNN` could support only a single batch dimension, i.e., inputs of the form `(batch, time, *features)`. Currently, it supports both multiple batch dimensions and multiple feature dimensions.
* Another approach could be a complete redesign of how Flax deals with recurrent states. For example, a `memory` collection could be handled as part of the variables. However, this introduces challenges such as handling stateless cells during training, passing state from one layer to another, and performing initialization inside `scan`.

### Refactor Cost
Initial TGP results showed 761 broken and 110 failed tests. However, after fixing one test, TGP results in 231 broken and 13 failed tests so there seems to be a lot
of overlap between the broken tests.

To minimize refactor costs, the current implementation will be kept for Google internal users under a deprecated name. This will allow users to migrate to the new API at their own pace. For Open Source users we should bump Flax version to
`0.7.0` so existing users can continue to depend on `0.6.x` versions.
