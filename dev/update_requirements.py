# Copyright 2020 The Flax Authors.
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

"""Utility script for updating examples' "requirements.txt"

WARNING: Updates examples/*/requirements.txt in place, so you probably want to
run this in a clean working directory...
"""

import glob
import pathlib
import re
import sys


import_re = re.compile(r'(?:from|import)\s+(\w+)')
pkg_map = {
  'absl': 'absl-py',
  'atari_py': 'atari-py',
  'cv2': 'opencv-python',
  'ml_collections': 'ml-collections',
  'PIL': 'Pillow',
  'tensorflow_datasets': 'tensorflow-datasets',
  'tensorflow_text': 'tensorflow-text',
}
standard_libs = set('codecs collections dataclasses datetime enum functools math multiprocessing itertools os pathlib random re sys tempfile time typing unicodedata warnings'.split(' '))


if len(sys.argv) != 2:
  print()
  print('Expected single argument that has form "clu-0.0.6 flax-0.3.6 ignoreme- ..."')
  print('Can be copied from "Install Dependencies" in latest build')
  print()
  print('https://github.com/google/flax/actions/workflows/build.yml')
  print()
  sys.exit(-1)

versions = {
    pkg_version[:pkg_version.rindex('-')]: pkg_version[pkg_version.rindex('-') + 1:]
    for pkg_version in sys.argv[1].split(' ')
    if '-' in pkg_version
}
# print(sorted(versions.items()))


examples_root = pathlib.Path(__file__).absolute().parents[1] / 'examples'

for directory in examples_root.glob('*'):
  if not directory.is_dir():
    continue
  requirements = directory / 'requirements.txt'
  if not requirements.exists():
    print(f'Skipping directory "{directory}"')
    continue

  pkgs = set()
  modules = set()
  for path in directory.glob('*'):
    if path.is_dir() and any(path.glob('*.py')):
      modules.add(path.parts[-1])
  for py in directory.glob('*.py'):
    modules.add(py.parts[-1][:-3])
    for line in open(py):
      m = import_re.match(line)
      if m:
        pkgs.add(m.group(1))

  pkgs -= modules | standard_libs

  print(f'{requirements} -', end=' ')
  with requirements.open('w') as f:
    for pkg in sorted(pkgs, key=str.casefold):
      pkg = pkg_map.get(pkg, pkg)
      print(f'{pkg}-{versions[pkg]}', end=' ')
      if not versions[pkg]:
        continue
      f.write(f'{pkg}=={versions[pkg]}\n')
    print()
