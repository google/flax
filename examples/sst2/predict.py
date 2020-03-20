# Copyright 2020 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Make predictions with a trained SST-2 model."""

@jax.jit
def predict_step(model: nn.Module, inputs: np.ndarray):
  logits = model(inputs, train=False)
  return get_predictions(logits)


def predict(model: nn.Module, test_ds: tf.data.Dataset):
  result = []
  rng = jax.random.PRNGKey(0)
  with nn.stochastic(rng):
    for ex in tfds.as_numpy(test_ds):
      inputs, labels = ex['sentence'], ex['label']
      predictions = predict_step(model, inputs)
      result += predictions.flatten().tolist()
  return np.array(result)


# Make test set predictions.
_, _, test_batches = get_batches(train_ds, valid_ds, test_ds, seed=0)
test_predictions = predict(best_model, test_batches)
test_predictions.shape

# Let's look at the predictions together with the original sentence.
num_examples = 10
for original, prediction in zip(
    tfds.as_numpy(data['test'].take(num_examples)), 
    test_predictions[:num_examples]):
  print('Sentence:  ', original['sentence'].decode('utf8'))
  print('Prediction:', 'positive' if prediction else 'negative')
  print()

