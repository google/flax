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

"""
Read all of the HOWTO .diff files and convert them into .html files
that are both Python syntax highlighted /and/ diff syntax highlighted.

Then these can be included directly in readthedocs as inline HTML
files.
"""

import pygments
import pygments.formatters
from pygments.lexers import PythonLexer

import re
import os

def main():
  print("Formatting HOWTOs into HTML files in _formatted_howtos/")
  os.makedirs('_formatted_howtos', exist_ok=True)
  for diff_filename in os.listdir(os.path.join('..', 'howtos', 'diffs')):
    format_howto(os.path.join('..', 'howtos', 'diffs', diff_filename),
           os.path.join('_formatted_howtos', diff_filename + '.html'))

def format_howto(input_file, output_file):
  # Load one of our HOWTO diff files.
  with open(input_file) as f:
    diff = f.readlines()

  # Ignore the first two line, which look like:
  #
  # diff --git a/examples/mnist/train.py b/examples/mnist/train.py
  # index 51d2fde..a9d7dcb 100644
  diff = diff[2:]

  # Remove double newlines of diff context from `diff` (which is a
  # list of lines).

  # Diff lines start with a charater in {'+', '-', ' '} designating
  # insert, remove or context.  Create a regexp that matches empty
  # lines that are either of these three types.
  empty_line_regexp = re.compile('[+\\- ]\n')
  diff = [diff[lineno] for lineno in range(len(diff))
      if lineno == 0 or not (
        empty_line_regexp.match(diff[lineno]) and
        empty_line_regexp.match(diff[lineno-1])
       )]

  # Don't do any special formatting.
  class RawHtmlFormatter(pygments.formatters.HtmlFormatter):
    def wrap(self, source, outfile):
      return source

  # Run `diff` through the normal pygments Python syntax
  # highlighter. Get back an array of HTML lines.
  colored_diff = (
    pygments.highlight('\n'.join(diff), PythonLexer(), RawHtmlFormatter())
  ).splitlines()

  # Write a newly formatted diff-and-Python syntax highlighted
  # output HTML file, that can be inline included into any .rst
  # file.
  with open(output_file, 'w') as out_file:
    # Add the relevant DIVs that Sphinx adds around code blocks.
    print('<div class="highlight-default notranslate"><div class="highlight">',
        file=out_file)
    print('<pre class="code">', file=out_file)

    for line in colored_diff:
      line = line.rstrip('\n')
      if line == '':
        continue

      if line.startswith('<span class="o">@@'):
        print('<span style="background-color: rgba(128, 128, 128, 0.3)">[...]</span>',
            file=out_file)
      elif line.startswith('<span class="o">+'):
        print('<span style="background-color: rgba(0, 255, 0, 0.3)">' + line + '</span>',
            file=out_file)
      elif line.startswith('<span class="o">-'):
        print('<span style="background-color: rgba(255, 0, 0, 0.3)">' + line + '</span>',
            file=out_file)
      else:
        print(line, file=out_file)

    print('</pre></div></div>', file=out_file)


main() 
