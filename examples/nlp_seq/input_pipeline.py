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

"""Input pipeline for the sequence tagging dataset."""

import codecs
import collections
import enum

import tensorflow as tf  # pytype: disable=import-error


# Values for padding, unknown words and a root.
PAD = '<p>'
PAD_ID = 0

UNKNOWN = '<u>'
UNKNOWN_ID = 1

ROOT = '<r>'
ROOT_ID = 2


class CoNLLAttributes(enum.Enum):
  """CoNLL attributre names and indices.

  A UD CoNLL file looks like:
  1    They     they    PRON    PRP    Case=Nom|Number=Plur       2    nsubj
  2    buy      buy     VERB    VBP    Number=Plur|PTense=Pres    0    root
  3    books    book    NOUN    NNS    Number=Plur                2    obj
  4    .        .       PUNCT   .      _                          2    punct

  For details, please see: http://universaldependencies.org/format.html.
  """
  ID = 0
  FORM = 1
  LEMMA = 2
  UPOS = 3
  XPOS = 4
  FEATS = 5
  HEAD = 6
  DEPREL = 7


def create_vocabs(filename, max_num_forms=100000):
  """Loads corpus and create vocabulary lists.

  Args:
    filename: file name of a corpus.
    max_num_forms: maximum number of tokens included.

  Returns:
    Dictionary containing named vocab dictionaries.

  """
  form_counter = collections.Counter()
  xpos_counter = collections.Counter()
  with tf.io.gfile.GFile(filename, 'rb') as f:
    for line in codecs.getreader('utf-8')(f):
      line = line.strip()
      split = line.split('\t')
      if not line.startswith('#') and split[0]:
        form_counter[split[CoNLLAttributes.FORM.value]] += 1
        xpos_counter[split[CoNLLAttributes.XPOS.value]] += 1

  special_tokens = {PAD: PAD_ID, UNKNOWN: UNKNOWN_ID, ROOT: ROOT_ID}

  # create word form vocab
  vocabs = {'forms': {}, 'xpos': {}}
  vocabs['forms'].update(special_tokens)
  vocabs['forms'].update({
      form[0]: id for id, form in enumerate(
          form_counter.most_common(max_num_forms), start=ROOT_ID + 1)
  })

  # create xpos vocab
  vocabs['xpos'].update(special_tokens)
  vocabs['xpos'].update({
      tag[0]: id
      for id, tag in enumerate(xpos_counter.most_common(), start=ROOT_ID + 1)
  })

  return vocabs


def create_token(token, attributes, vocabs):
  """Map for a token a selected subset of attributes to indices.

  Input example: CoNLL 09 representation for a token.
    ['Ms.', 'ms.', 'ms.', 'NNP', '_', '2', 'TITLE]
  Output example: Indices as defined in self._attributes, e.g., [word form,
    part-of-speech tag, and head].
    [1025, 3, 1]

  Args:
    token: CoNLL token atrributes.
    attributes: selected attributes.
    vocabs: dictonery of vocabs.

  Returns:
    List of attribute ids for a token, e.g. [1025, 3] with word id and pos id.

  Raises:
    ValueError: CoNLL attribute requested but not covered by mapping.
  """
  selected_attributes = []
  for attribute in attributes:
    index = attribute.value
    if attribute == CoNLLAttributes.FORM:
      selected_attributes.append(vocabs['forms'].get(token[index], UNKNOWN_ID))
    elif attribute == CoNLLAttributes.XPOS:
      selected_attributes.append(vocabs['xpos'].get(token[index], UNKNOWN_ID))
    elif attribute == CoNLLAttributes.HEAD:
      selected_attributes.append(int(token[index]))
    else:
      raise ValueError('CoNLL index %s not covered by mapping.' %
                       str(attribute.name))
  return selected_attributes


def create_sentence_with_root(attributes, vocabs):
  """Create a sentence containing a root.

  Args:
    attributes: attributes extracted from token.
    vocabs: dictonery of vocabs.

  Returns:
    A list representing a sentence containing the root only,
    e.g., [[2, 1, 0]] for root word, unknown xpos, and head 0.
  """
  # Create the token properties of an artificial root node.
  token_properties = [ROOT for _ in range(12)]  # CoNLL 09 has 12 columns.
  token_properties[CoNLLAttributes.ID.value] = '0'
  token_properties[CoNLLAttributes.HEAD.value] = '0'
  token = create_token(token_properties, attributes, vocabs)
  if len(token) == 1:
    token = token[0]
  return [token]


def sentences_from_conll_data(corpus_filename,
                              vocabs,
                              attributes,
                              max_sentence_length=1000):
  """Load and returns conll data in list format.

  Args:
    corpus_filename: filename of corpus.
    vocabs: dictionary of vocabs
    attributes: list of conll attributes to include into the batch
    max_sentence_length: cut off sentences longer as max tokens

  Yields:
      A sentence as a list of tokens while tokens are lists of attributes.
  """
  with tf.io.gfile.GFile(corpus_filename, 'rb') as f:
    sentence = create_sentence_with_root(attributes, vocabs)
    for line in codecs.getreader('utf-8')(f):
      line = line.strip()
      if line.startswith('#'):
        continue
      split = line.split('\t')
      if split[0]:  # Not an empty line, process next token:
        if len(sentence) < max_sentence_length:
          if len(attributes) == 1:
            sentence.append(create_token(split, attributes, vocabs)[0])
          else:
            sentence.append(create_token(split, attributes, vocabs))
      else:  # Sentences start with an empty line, yield sentence:
        yield sentence

        # Reset sentence.
        sentence = create_sentence_with_root(attributes, vocabs)
    if len(sentence) > 1:  # sentences does not only contain a root.
      yield sentence


def sentence_dataset_dict(filename,
                          vocabs,
                          attributes_input,
                          attributes_target,
                          batch_size,
                          bucket_size,
                          repeat=None,
                          prefetch_size=tf.data.experimental.AUTOTUNE):
  """Combines sentences into a dataset of padded batches.

  Args:
    filename: file name of a corpus.
    vocabs: dictionary of dictionaries to map from strings to ids.
    attributes_input: attributes for the input.
    attributes_target: target attributes empty targets is not inclueded.
    batch_size: the size of a batch.
    bucket_size: the size of a bucket.
    repeat: number of times the dataset is repeated.
    prefetch_size: prefetch size of the data.

  Returns:
    Returns dataset as dictionary containing the data as key value pairs.
  """
  data_keys = ['inputs']
  if attributes_target:
    data_keys.append('targets')

  def generator():
    """Generator to create the data."""
    input_generator = sentences_from_conll_data(
        filename, vocabs, attributes_input, max_sentence_length=bucket_size)

    if attributes_target:
      target_generator = sentences_from_conll_data(
          filename, vocabs, attributes_target, max_sentence_length=bucket_size)

    for inputs in input_generator:
      data = {'inputs': inputs}
      if attributes_target:
        data['targets'] = next(target_generator)
      yield data

  output_types = {k: tf.float32 for k in data_keys}
  output_shapes = {k: (None,) for k in data_keys}
  dataset = tf.data.Dataset.from_generator(
      generator, output_types=output_types, output_shapes=output_shapes)

  # cache the dataset in memory and repeat.
  dataset = dataset.cache()
  dataset = dataset.repeat(repeat)

  # static padding up to bucket size.
  padded_shapes = {k: [bucket_size] for k in data_keys}
  dataset = dataset.padded_batch(
      batch_size=batch_size, padded_shapes=(padded_shapes))

  dataset = dataset.prefetch(prefetch_size)
  return dataset
