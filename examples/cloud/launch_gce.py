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

# Script for creating VM on Google cloud and run a Flax example inside.
# See ./README.md for instructions.

import datetime
import os
import re
import subprocess
import time
from typing import Sequence

from absl import app
from absl import flags

# General options.
flags.DEFINE_bool(
    'dry_run',
    False,
    help='If set, then the command to launch the GCE instance will only be '
    'printed to stdout.')
flags.DEFINE_bool(
    'connect',
    False,
    help='Same as --wait, but directly connect to VM once it is ready.')
flags.DEFINE_bool(
    'wait', False,
    help='If set, then the script will wait until VM is ready. If VM_READY_CMD '
    'is set in environment, then that command will be executed once the VM '
    'is ready. Useful for sending a notification, e.g. "osascript" (mac).')

# Machine configuration.
flags.DEFINE_string('project', None, help='Name of the Google Cloud project.')
flags.DEFINE_string('zone', None, help='Zone in which the VM will be created.')
flags.DEFINE_string(
    'machine_type',
    None,
    help='Machine type to use for VM. See "gcloud compute machine-types list".')
flags.DEFINE_string(
    'accelerator_type',
    '',
    help='Type of accelerator to use, or empty. '
    'See "gcloud compute accelerator-types list".'
)
flags.DEFINE_integer(
    'shutdown_secs',
    300,
    help='How long to wait (after successful/failed training) before shutting '
    'down the VM. Set to 0 to disable.'
)
flags.DEFINE_integer(
    'accelerator_count', 8, help='Number of accelerators to use.')

# GCS configuration.
flags.DEFINE_string(
    'gcs_workdir_base',
    None,
    help='GCS base directory for model output. The --workdir argument will be '
    'constructed from {gcs_workdir_base}/{example}/{name}/{timestamp} .')
flags.DEFINE_string(
    'tfds_data_dir',
    '',
    help='Optional tfds data directory. This can be useful to prepare datasets '
    'on GCS and then point the jobs to this preloaded directory. Dataset will '
    'be downloaded from the web if not specified.')

# Repo configuration.
flags.DEFINE_string(
    'repo', 'https://github.com/google/flax', help='Git repository')
flags.DEFINE_string('branch', 'main', help='Git repository')

# Example configuration.
flags.DEFINE_string(
    'example', None, help='Name of Flax example (e.g. "imagenet").')
flags.DEFINE_string(
    'args',
    '',
    help='Any additional command line arguments for {example}_main.py, like '
    'for example --config. Note that --workdir will be provided by the '
    'script.')

# Run configuration.
flags.DEFINE_string(
    'name',
    None,
    help='Name of the experiment. Note that the provided name will be '
    'extended to {example}/{name}/{timestamp}')

FLAGS = flags.FLAGS
flags.mark_flags_as_required(
    ['project', 'zone', 'machine_type', 'gcs_workdir_base', 'example', 'name'])

timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


def generate_startup_file(vm_name: str) -> str:
  directory = os.path.dirname(os.path.abspath(__file__))
  startup_script_src = os.path.join(directory, 'startup_script.sh')
  startup_script_dst = os.path.join(directory, f'{vm_name}-startup_script.sh')
  assert not os.path.exists(startup_script_dst)
  with open(startup_script_src, encoding='utf8') as f:
    startup_script_content = f.read()
  for from_str, to_str in (
      ('__REPO__', FLAGS.repo),
      ('__BRANCH__', FLAGS.branch),
      ('__EXAMPLE__', FLAGS.example),
      ('__TIMESTAMP__', timestamp),
      ('__NAME__', FLAGS.name),
      ('__ARGS__', FLAGS.args),
      ('__GCS_WORKDIR_BASE__', FLAGS.gcs_workdir_base),
      ('__TFDS_DATA_DIR__', FLAGS.tfds_data_dir),
      ('__ACCELERATOR_TYPE__', FLAGS.accelerator_type),
      ('__SHUTDOWN_SECS__', str(FLAGS.shutdown_secs))
  ):
    startup_script_content = startup_script_content.replace(from_str, to_str)
  with open(startup_script_dst, 'w', encoding='utf8') as f:
    f.write(startup_script_content)
  return startup_script_dst


def launch_gce(*, vm_name: str, startup_script: str):
  # Note : Use `gcloud compute images list --project ml-images` to get a list
  # of available VM images.
  args = [
      'gcloud', 'compute', 'instances', 'create', vm_name,
      f'--project={FLAGS.project}', f'--zone={FLAGS.zone}',
      '--image=c1-deeplearning-tf-2-10-cu113-v20221107-debian-10',
      '--image-project=ml-images', f'--machine-type={FLAGS.machine_type}',
      '--scopes=cloud-platform,storage-full', '--boot-disk-size=256GB',
      '--boot-disk-type=pd-ssd', '--metadata=install-nvidia-driver=True',
      f'--metadata-from-file=startup-script={startup_script}'
  ]
  if FLAGS.accelerator_type and FLAGS.accelerator_count:
    args.extend([
        '--maintenance-policy=TERMINATE',
        f'--accelerator=type={FLAGS.accelerator_type},count={FLAGS.accelerator_count}',
    ])

  if FLAGS.dry_run:
    print()
    print('Would run the following command without --dry-run:')
    print()
    print(' \\\n    '.join(args))
    print()
    return

  print()
  print('Creating instance on GCE... This will take some minutes...')
  print()
  result = subprocess.run(args)
  if result.returncode:
    raise RuntimeError('Could not create VM!')


def print_howto(login_args: Sequence[str]):
  print(f'''
###############################################################################
###############################################################################

You can start/stop the instace via the web UI:
https://console.cloud.google.com/compute/instances?project={FLAGS.project}

Once the VM has started, you can login and connect to the training session:

{' '.join(login_args)}

Note that you can disconnect from the tmux session without stopping the training
with the keystrokes 'CTRL-B A'. See "man tmux" for help about tmux.

To observe the training via Tensorboard, simply run in your local computer:

$ tensorboard --logdir={FLAGS.gcs_workdir_base}

You can also browse the files at

https://console.cloud.google.com/storage/browser/{FLAGS.gcs_workdir_base.replace('gs://', '')}

###############################################################################
###############################################################################
''')


def main(_):
  for name in ('repo', 'branch', 'example', 'name', 'gcs_workdir_base'):
    value = getattr(FLAGS, name)
    if re.match(r'[^\w:/_-]', value):
      raise ValueError(f'Invalid flag value: --{name}="{value}"')
  example_base_directory = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      os.path.pardir,
  )
  if not os.path.isdir(os.path.join(example_base_directory, FLAGS.example)):
    raise ValueError(f'Could not find --example={FLAGS.example}')
  if FLAGS.connect and FLAGS.dry_run:
    raise ValueError('Cannot --connect to VM with --dry_run')

  vm_name = '-'.join([
      'flax',
      FLAGS.example,
      timestamp,
  ])
  vm_name = re.sub(r'[^a-z0-9-]', '-', vm_name)

  startup_script = generate_startup_file(vm_name)
  launch_gce(vm_name=vm_name, startup_script=startup_script)

  login_args = [
      'gcloud',
      'compute',
      'ssh',
      '--project',
      FLAGS.project,
      '--zone',
      FLAGS.zone,
      vm_name,
      '--',
      '/sudo_tmux_a.sh',
  ]

  print('Your instance is being started...')

  print_howto(login_args)

  if FLAGS.connect or FLAGS.wait:
    login_true_args = login_args[:-1] + ['true']
    while True:
      try:
        result = subprocess.run(
            login_true_args, timeout=10, stderr=subprocess.PIPE)
        if result.returncode == 0:
          break
        stderr = result.stderr.decode('utf8')
        if 'connection refused' in stderr.lower():
          print('(Connection refused - waiting a little longer...)')
          time.sleep(20)
        else:
          raise ValueError(f'Unknown error: {stderr}')
      except ValueError as e:
        if 'HTTP 502' not in str(e):
          raise e
        print('(Bad Gateway - waiting a little longer...)')
        time.sleep(20)
      except subprocess.TimeoutExpired:
        print('(Timeout - waiting a little longer...)')
        time.sleep(20)
    if 'VM_READY_CMD' in os.environ:
      os.system(os.environ['VM_READY_CMD'])
    if FLAGS.connect:
      result = subprocess.run(login_args)
      # SSH session has cleared previous message, print it again.
      print_howto(login_args)


if __name__ == '__main__':
  app.run(main)
