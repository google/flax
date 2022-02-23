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

"""Utility script for updating examples' "requirements.txt".

WARNING: Updates examples/*/requirements.txt in place, so you probably want to
run this in a clean working directory...

This script is useful to automatically update the requirements of the individual
examples. It's desirable to have these dependencies fully specified and
guaranteed to be complete in order to download an example directly from Github
and run it on a virtual machine without any additional manual package
installation (note though that jax/jaxlib must already be installed).

To update all "requirements.txt" the program is run with a `--versions` argument
that specifies the version of every package that is required by any of the
examples. If any of the versions is not specified, the program will fail with an
error.

Example invocations (from the Flax repo root directory):

python dev --versions='absl-py-0.12.0 clu-0.0.6 flax-0.3.6'

The actual pypi-version list can be read copied from the "Install Dependencies"
step from the latest Github build action:
https://github.com/google/flax/actions/workflows/build.yml

Alternatively, the list can also be provided from the local environment with:

python dev --versions="$(pip freeze | sed s/==/-/g) flax-0.3.6"
"""

import pathlib
import re

from absl import app
from absl import flags


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'versions',
    None,
    'space-separated list of pkg-name-1.0.0 pairs. can be obtained from local '
    'environment with '
    '`--version="$(pip freeze | sed s/==/-/g) flax-0.3.6"` '
    '(note the flax version "override") '
    'or from the "install dependencies" step in the github build action '
    'https://github.com/google/flax/actions/workflows/build.yml')
flags.mark_flag_as_required('versions')
flags.DEFINE_bool('verbose', False, 'enables verbose output.')
flags.DEFINE_list('ignore', ['jax'], 'packages not to add to requirements.')


import_re = re.compile(r'(?:from|import)\s+(\w+)')
# maps `import cv2` to `pip install opencv-python`
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


def main(argv):
  del argv

  versions = {
      pkg_version[:pkg_version.rindex('-')]: pkg_version[pkg_version.rindex('-') + 1:]
      for pkg_version in FLAGS.versions.replace('\n', ' ').split(' ')
      if '-' in pkg_version
  }
  if FLAGS.verbose:
    print('parsed versions:', sorted(versions.items()))
  ignore = set(FLAGS.ignore)

  examples_root = pathlib.Path(__file__).absolute().parents[1] / 'examples'

  for example_dir in examples_root.glob('*'):
    if not example_dir.is_dir():
      continue
    requirements = example_dir / 'requirements.txt'
    if not requirements.exists():
      print(f'Skipping directory "{example_dir}"')
      continue

    pkgs = set()
    local_pkgs_and_modules = set()
    for path in example_dir.glob('*'):
      if path.is_dir() and any(path.glob('*.py')):
        local_pkgs_and_modules.add(path.parts[-1])  # local package
    for py in example_dir.glob('*.py'):
      local_pkgs_and_modules.add(py.parts[-1][:-3])  # local module
      for line in open(py):
        m = import_re.match(line)
        if m:
          pkgs.add(m.group(1))

    pkgs -= local_pkgs_and_modules | standard_libs

    print(f'{requirements} -', end=' ')
    with requirements.open('w') as f:
      for pkg in sorted(pkgs, key=str.casefold):
        if pkg in ignore: continue
        pkg = pkg_map.get(pkg, pkg)
        print(f'{pkg}-{versions[pkg]}', end=' ')
        f.write(f'{pkg}=={versions[pkg]}\n')
      print()


if __name__ == '__main__':
  app.run(main)
