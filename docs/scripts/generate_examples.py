# Copyright 2020 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2020 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil

def main():
  print('Copying example READMEs in examples/ and generating docs')
  src_base = os.path.join('..', 'examples')
  dst_base = '_examples'
  # shutil.copytree(src, dst)

  # This is brittle, but we want to be careful about what we bring into docs
  # for processing (e.g., notebooks can be problematic)
  exts = ['.md', '.png']
  doc_name = 'README.md'

  for src_dir, dirs, files in os.walk(src_base):
    # Assumes documentation is a README.md file
    dst_dir = src_dir.replace(src_base, dst_base)
    os.makedirs(dst_dir, exist_ok=True)

    for file in files:
      name, ext = os.path.splitext(file)
      if ext in exts:
        shutil.copyfile(os.path.join(src_dir, file),
                        os.path.join(dst_dir, file))

if __name__ == 'main':
    main()
