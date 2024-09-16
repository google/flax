# Copyright 2024 The Flax Authors.
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

"""Sphinx directive for creating code diff tables.

Use directive as follows:

.. codediff::
  :title: <LEFT_CODE_BLOCK_TITLE>, <RIGHT_CODE_BLOCK_TITLE>

  <CODE_BLOCK_LEFT>
  ---
  <CODE_BLOCK_RIGHT>

In order to highlight a line of code, append "#!" to it.
"""


import sphinx
from docutils import nodes
from docutils.parsers.rst import directives
from docutils.statemachine import ViewList
from sphinx.util.docutils import SphinxDirective

MISSING = object()


class CodeDiffParser:
  def parse(
    self,
    lines: list[str],
    title: str,
    groups: list[str] | None = None,
    skip_test: str | None = None,
    code_sep: str = '---',
    sync: object = MISSING,
  ):
    """Parse the code diff block and format it so that it
    renders in different tabs and is tested by doctest.

    For example:

      .. testcode:: tab0, tab2, tab3

        <CODE_BLOCK_A>

      .. codediff::
        :title: Tab 0, Tab 1, Tab 2, Tab 3
        :groups: tab0, tab1, tab2, tab3
        :skip_test: tab1, tab3

        <CODE_BLOCK_B0>

        ---

        <CODE_BLOCK_B1>

        ---

        <CODE_BLOCK_B2>

        ---

        <CODE_BLOCK_B3>

    For group tab0: <CODE_BLOCK_A> and <CODE_BLOCK_B0> are executed.
    For group tab1: Nothing is executed.
    For group tab2: <CODE_BLOCK_A> and <CODE_BLOCK_B2> are executed.
    For group tab3: <CODE_BLOCK_A> is executed.

    Arguments:
      lines: a string list, where each element is a single string code line
      title: a single string that contains the titles of each tab (they should
        be separated by commas)
      groups: a single string that contains the group of each tab (they should
        be separated by commas). Code snippets that are part of the same group
        will be executed together. If groups=None, then the group names will
        default to the tab title names.
      skip_test: a single string denoting which group(s) to skip testing (they
        should be separated by commas). This is useful for legacy code snippets
        that no longer run correctly anymore. If skip_test=None, then no tests
        are skipped.
      code_sep: the separator character(s) used to denote a separate code block
        for a new tab. The default code separator is '---'.
      sync: an option for Sphinx directives, that will sync all tabs together.
        This means that if the user clicks to switch to another tab, all tabs
        will switch to the new tab.
    """
    titles = [t.strip() for t in title.split(',')]
    num_tabs = len(titles)

    sync = sync is not MISSING
    # skip legacy code snippets in upgrade guides
    if skip_test is not None:
      skip_tests = {index.strip() for index in skip_test.split(',')}
    else:
      skip_tests = set()

    code_blocks = '\n'.join(lines)
    if code_blocks.count(code_sep) != num_tabs - 1:
      raise ValueError(
        f'Expected {num_tabs-1} code separator(s) for {num_tabs} tab(s), but got {code_blocks.count(code_sep)} code separator(s) instead.'
      )
    code_blocks = [
      code_block.split('\n')
      for code_block in code_blocks.split(code_sep + '\n')
    ]  # list[code_tab_list1[string_line1, ...], ...]

    # by default, put each code snippet in a different group denoted by an index number, to be executed separately
    if groups is not None:
      groups = [group_name.strip() for group_name in groups.split(',')]
    else:
      groups = titles
    if len(groups) != num_tabs:
      raise ValueError(
        f'Expected {num_tabs} group assignment(s) for {num_tabs} tab(s), but got {len(groups)} group assignment(s) instead.'
      )

    tabs = []
    test_codes = []
    for i, code_block in enumerate(code_blocks):
      if groups[i] not in skip_tests:
        test_codes.append((code_block, groups[i]))
      tabs.append((titles[i], self._code_block(code_block)))
    output = self._tabs(*tabs, sync=sync)

    return output, test_codes

  def _code_block(self, lines):
    """Creates a codeblock."""
    # Remove right trailing whitespace so we can detect the comments.
    lines = [x.rstrip() for x in lines]
    highlight = lambda x: x.endswith('#!')
    code = map(lambda x: x[:-2].rstrip() if highlight(x) else x, lines)
    highlights = [i + 1 for i in range(len(lines)) if highlight(lines[i])]
    highlights = ','.join(str(i) for i in highlights)

    directive = ['.. code-block:: python']
    if highlights:
      directive += [f'  :emphasize-lines: {highlights}']

    # Indent code and add empty line so the code is picked up by the directive.
    return directive + [''] + list(map(lambda x: '  ' + x, code))

  def _tabs(self, *contents: tuple[str, list[str]], sync):
    output = ['.. tab-set::'] + ['  ']

    for title, content in contents:
      output += [f'  .. tab-item:: {title}']

      if sync:
        key = title.strip()
        output += [f'    :sync: {key}']

      output += ['    ']
      output += ['    ' + line for line in content]

    return output


class CodeDiffDirective(SphinxDirective):
  has_content = True
  option_spec = {
    'title': directives.unchanged,
    'groups': directives.unchanged,
    'skip_test': directives.unchanged,
    'code_sep': directives.unchanged,
    'sync': directives.flag,
  }

  def run(self):
    table_code, test_codes = CodeDiffParser().parse(
      list(self.content), **self.options
    )

    # Create a test node as a comment node so it won't show up in the docs.
    # We add attribute "testnodetype" so it is be picked up by the doctest
    # builder. This functionality is not officially documented but can be found
    # in the source code:
    # https://github.com/sphinx-doc/sphinx/blob/master/sphinx/ext/doctest.py
    # (search for 'testnodetype').
    test_nodes = []
    for test_code, group in test_codes:
      test_node = nodes.comment(
        '\n'.join(test_code),
        '\n'.join(test_code),
        testnodetype='testcode',
        groups=[group],
      )
      self.set_source_info(test_node)
      test_node['options'] = {}
      test_node['language'] = 'python3'
      test_nodes.append(test_node)

    # The table node is the side-by-side diff view that will be shown on RTD.
    table_node = nodes.paragraph()
    self.content = ViewList(table_code, self.content.parent)
    self.state.nested_parse(self.content, self.content_offset, table_node)

    return [table_node] + test_nodes


def setup(app):
  app.add_directive('codediff', CodeDiffDirective)

  return {
    'version': sphinx.__display_version__,
    'parallel_read_safe': True,
    'parallel_write_safe': True,
  }
