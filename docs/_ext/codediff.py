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
from typing import Optional, Sequence
import itertools

from docutils import nodes
from docutils.parsers.rst import directives
from docutils.statemachine import ViewList

import sphinx
from sphinx.util.docutils import SphinxDirective
"""Sphinx directive for creating code diff tables.

Use directive as follows:

.. codediff::
  :title_left: <LEFT_CODE_BLOCK_TITLE>
  :title_right: <RIGHT_CODE_BLOCK_TITLE>
  
  <CODE_BLOCK_LEFT>
  ---
  <CODE_BLOCK_RIGHT>

In order to highlight a line of code, prepend it with "#!".
"""

class CodeDiffParser:
  def parse(self, lines, title_left='Base', title_right='Diff', code_sep='---'):
    if code_sep not in lines:
      raise ValueError('Code separator not found! Code snippets should be '
                       f'separated by {code_sep}.')
    idx = lines.index(code_sep)
    code_left = self._code_block(lines[0: idx])
    test_code = lines[idx+1:]
    code_right = self._code_block(test_code)
    
    self.max_left = max(len(x) for x in code_left + [title_left])
    self.max_right = max(len(x) for x in code_right + [title_right])

    output = [
      self._hline(),
      self._table_row(title_left, title_right),
      self._hline(),
    ]

    for l, r in itertools.zip_longest(code_left, code_right, fillvalue=''):
      output += [self._table_row(l, r)]

    return output + [self._hline()], test_code

  def _code_block(self, lines):
    # Remove right trailing whitespace so we can detect the comments.
    lines = [x.rstrip() for x in lines]
    highlight = lambda x : x.endswith('#!')
    code = map(lambda x : x[:-2].rstrip() if highlight(x) else x, lines)
    highlights = [i+1 for i in range(len(lines)) if highlight(lines[i])]
    highlights = ','.join(str(i) for i in highlights)

    directive = ['.. code-block:: python']
    if highlights:
      directive += [f'  :emphasize-lines: {highlights}']

    # Indent code and add empty line so the code is picked up by the directive.
    return directive + [''] + list(map(lambda x: '  ' + x, code))

  def _hline(self):
    return '+' + '-'*(self.max_left+2) + '+' + '-'*(self.max_right+2) + '+'

  def _rfill(self, text, max_len):
    return text + ' ' * (max_len-len(text))

  def _table_row(self, left, right):
    text_left = self._rfill(left, self.max_left)
    text_right = self._rfill(right, self.max_right)
    return '| ' + text_left + ' | ' + text_right + ' |'


class CodeDiffDirective(SphinxDirective):
  has_content = True
  option_spec = {
    'title_left': directives.unchanged,
    'title_right': directives.unchanged,
    'code_sep': directives.unchanged,
    }

  def run(self):
    table_code, test_code = CodeDiffParser().parse(list(self.content), **self.options)

    # Create a test node as a comment node so it won't show up in the docs.
    # We add attribute "testnodetype" so it is be picked up by the doctest
    # builder.
    test_code = '\n'.join(test_code)
    test_node = nodes.comment(test_code, test_code, testnodetype='testcode')
    # Set the source info so the error message is correct when testing.
    self.set_source_info(test_node)
    test_node['language'] = 'python3'

    # The table node is the side-by-side diff view that will be shown on RTD.    
    table_node = nodes.paragraph()
    self.content = ViewList(table_code, self.content.parent)
    self.state.nested_parse(self.content, self.content_offset, table_node)
    
    return [table_node, test_node]

def setup(app):
    app.add_directive('codediff', CodeDiffDirective)

    return {
        'version': sphinx.__display_version__,
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
