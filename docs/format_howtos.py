"""
Read all of the HOWTO .diff files and convert them into .html files
that are both Python syntax highlighted /and/ diff syntax highlighted.

Then these can be included directly in readthedocs as inline HTML
files.
"""

import pygments
import pygments.formatters
from pygments.lexers import PythonLexer

import os

def main():
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

  # Remove more than one empty line in a row
  diff = [diff[lineno] for lineno in range(len(diff))
      if lineno == 0 or not (
        diff[lineno].rstrip(' \n') == '' and
        diff[lineno-1].rstrip(' \n') == ''
       )]

  # Don't do any special formatting
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
    # Add the relevant DIVs that Sphinx adds around code blocks
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
