## Tensorflow Datasets Test Metadata files

Examples download and use datasets using Tensorflow Dataset (TFDS). 

Each example should ideally have a test which executes the training and evaluation loop atleast once (for code coverage) with tests taking as minimal time as possible. 
For input, we could either:
- Generate and use fake data. This might lead to tests skipping execution of input pipeline.
- Download and use the original dataset. This might become impossible for examples like Imagenet which requires ~180GB RAM.

TFDS supports mocking calls to TFDS and generates random data instead using [tfds.testing.mock_data](https://www.tensorflow.org/datasets/api_docs/python/tfds/testing/mock_data).
Although this requires original/true metadata files (dataset_info.json, label.txt, vocabulary files etc) to determine the dtype, shape, potential input encodings etc of the 
random data being generated.

In order to make `mock_data` work for Flax examples, follow the below instructions:
- Copy the metadata files for the dataset being mocked from [here](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/testing/metadata) and paste them in `flax/examples/testing/datasets`.
- In the test file, wrap the code which loads dataset using TFDS with `tfds.testing.mock_data` context manager.

Sample code:
```python
# Number of train and eval examples to load.
num_examples=8 

# testing/datasets dir.
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
data_dir = parent_dir + '/testing/datasets'

with tfds.testing.mock_data(num_examples=num_examples, data_dir=data_dir):
  # code executing tfds.load() or equivalent.
```

Example test: `flax/examples/imagenet_lib_test.py` 