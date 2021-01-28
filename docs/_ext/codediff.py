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
import dataclasses
from typing import Sequence
import itertools

from docutils import nodes
from docutils.parsers.rst import directives
from docutils.statemachine import ViewList

import sphinx
from sphinx.util.docutils import SphinxDirective
"""Sphinx directive for creating code diff tables.

Use directive as follows:

.. codediff::
  :title-left: <LEFT_CODE_BLOCK_TITLE>
  :title-right: <RIGHT_CODE_BLOCK_TITLE>
  :highlight-left: <LINES_TO_HIGHLIGHT_LEFT>
  :highlight-right: <LINES_TO_HIGHLIGHT_RIGHT>
  
  <CODE_BLOCK_LEFT>
  ---
  <CODE_BLOCK_RIGHT>
"""

@dataclasses.dataclass
class CodeDiffBlock:
  lines: Sequence[str]
  title_left: str = 'Base code'
  title_right: str = 'Diff code'
  highlight_left: str = ''
  highlight_right: str = ''
  code_sep: str = '---'

  def __post_init__(self):
    if self.code_sep not in self.lines:
      raise ValueError('Code separator not found! Code snippets should be '
                       f'separated by {self.code_sep}.')
  

class CodeDiffParser:
  def __init__(self, code_block: CodeDiffBlock):
    self.code_block = code_block

  def parse(self):
    code = self.code_block
    idx = code.lines.index(code.code_sep)
    self.left_code, self.left_max = self._code_block(code.highlight_left, 0, idx)
    self.right_code, self.right_max = self._code_block(code.highlight_right, idx+1)
    
    output = [
      self._horizontal_line(),
      self._table_row(self.code_block.title_left, self.code_block.title_right),
      self._horizontal_line(),
    ]

    for left, right in itertools.zip_longest(self.left_code, self.right_code, fillvalue=''):
      output += [self._table_row(left, right)]

    return output + [self._horizontal_line()]

  def _code_block(self, highlights, start_idx, end_idx=None):
    lines = ['.. code-block:: python']
    if highlights:
      lines += [f'  :emphasize-lines: {highlights}']

    indent = lambda lines: ['  ' + line for line in lines]

    # Prefix empty line to code blocks to separate them from the code directive.
    lines += [''] + indent(self.code_block.lines[start_idx: end_idx])
    return lines, max(len(x) for x in lines)

  def _horizontal_line(self):
    return '+' + '-'*(self.left_max+2) + '+' + '-'*(self.right_max+2) + '+'

  def _rfill(self, text, max_len):
    return text + ' ' * (max_len-len(text))

  def _table_row(self, left, right):
    return '| ' + self._rfill(left, self.left_max) + ' | ' + self._rfill(right, self.right_max) + ' |'


class CodeDiffDirective(SphinxDirective):
  has_content = True
  option_spec = {
    'title_left': directives.unchanged,
    'title_right': directives.unchanged,
    'highlight_left': directives.unchanged,
    'highlight_right': directives.unchanged
    }

  def run(self):    
    code_diff = CodeDiffBlock(lines=list(self.content), **self.options)
    new_content = CodeDiffParser(code_diff).parse()

    node = nodes.paragraph()
    self.content = ViewList(new_content, self.content.parent)
    self.state.nested_parse(self.content, self.content_offset, node)
    return [node]

def setup(app):
    app.add_directive('codediff', CodeDiffDirective)

    return {
        'version': sphinx.__display_version__,
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
