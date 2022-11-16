# Copyright 2022 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for flax.examples.nlp.input_pipeline."""

import os

from absl.testing import absltest
import jax
import tensorflow as tf

import input_pipeline


# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


CONLL_DATA = """1\tThey\tthey\tPRON\tPRP\tCase=Nom|Number=Plur\t2\tnsubj
2\tbuy\tbuy\t VERB\tVBP\tNumber=Plur|PTense=Pres\t0\troot
3\tbooks\tbook\tNOUN\tNNS\tNumber=Plur\t2\tobj
4\t.\t.\tPUNCT\t.\t_\t2\tpunct

1\tThey\tthey\tPRON\tPRP\tCase=Nom|Number=Plur\t2\tnsubj
2\tbuy\tbuy\t VERB\tVBP\tNumber=Plur|PTense=Pres\t0\troot
3\tbooks\tbook\tNOUN\tNNS\tNumber=Plur\t2\tobj
4\t.\t.\tPUNCT\t.\t_\t2\tpunct

1\tNY\tNY\tNOUN\tNNS\tNumber=Singular\t0\troot
"""


class InputPipelineTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.test_tmpdir = self.create_tempdir()

    # Write a sample corpus.
    self._filename = os.path.join(self.test_tmpdir.full_path, 'data.conll')
    with tf.io.gfile.GFile(self._filename, 'w') as f:
      # The CoNLL data has to end with an empty line.
      f.write(CONLL_DATA)
      f.write('\n')

  def test_vocab_creation(self):
    """Tests the creation of the vocab."""
    vocabs = input_pipeline.create_vocabs(self._filename)
    self.assertEqual(
        vocabs['forms'], {
            '<p>': 0,
            '<u>': 1,
            '<r>': 2,
            'They': 3,
            'buy': 4,
            'books': 5,
            '.': 6,
            'NY': 7,
        })

  def testInputBatch(self):
    """Test the batching of the dataset."""
    vocabs = input_pipeline.create_vocabs(self._filename)

    attributes_input = [input_pipeline.CoNLLAttributes.FORM]
    attributes_target = []  # empty target for tagging of unlabeled data.
    sentence_dataset = input_pipeline.sentence_dataset_dict(
        self._filename, vocabs, attributes_input, attributes_target,
        batch_size=2, bucket_size=10, repeat=1)

    sentence_dataset_iter = iter(sentence_dataset)

    batch = next(sentence_dataset_iter)
    inputs = batch['inputs'].numpy().tolist()
    self.assertSameStructure(inputs, [[2., 3., 4., 5., 6., 0., 0., 0., 0., 0.],
                                      [2., 3., 4., 5., 6., 0., 0., 0., 0., 0.]])
    self.assertLen(batch, 1)  # make sure target is not included.

  def testInputTargetBatch(self):
    """Test the batching of the dataset."""
    vocabs = input_pipeline.create_vocabs(self._filename)

    attributes_input = [input_pipeline.CoNLLAttributes.FORM]
    attributes_target = [input_pipeline.CoNLLAttributes.XPOS]
    sentence_dataset = input_pipeline.sentence_dataset_dict(
        self._filename, vocabs, attributes_input, attributes_target,
        batch_size=2, bucket_size=10, repeat=1)

    sentence_dataset_iter = iter(sentence_dataset)

    batch = next(sentence_dataset_iter)
    inputs = batch['inputs'].numpy().tolist()
    self.assertSameStructure(inputs, [[2., 3., 4., 5., 6., 0., 0., 0., 0., 0.],
                                      [2., 3., 4., 5., 6., 0., 0., 0., 0., 0.]])
    targets = batch['targets'].numpy().tolist()
    self.assertSameStructure(targets,
                             [[2., 4., 5., 3., 6., 0., 0., 0., 0., 0.],
                              [2., 4., 5., 3., 6., 0., 0., 0., 0., 0.]])


if __name__ == '__main__':
  absltest.main()
