import datetime
import re
import subprocess

from absl import app
from absl import flags


FLAGS = flags.FLAGS

flags.DEFINE_string('project', help='Name of the Google Cloud project.')
flags.DEFINE_string('vm', help='Name of the VM to be created.')
flags.DEFINE_string('zone', help='Zone in which the VM will be created.')

flags.DEFINE_string('repo', help='Git repository')
flags.DEFINE_string('branch', default='master', help='Git repository')

flags.DEFINE_string('example', help='Flax example')
flags.DEFINE_string('config', help='Config (relative to example directory)')

flags.DEFINE_string('id', default=None)

timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


def generate_startup_file():
  pass

def launch_gce(startup_file):
  vm_name = re.sub(r'[^a-z0-9-]', '-'.join([
    'flax',
    FLAGS.example,
    timestamp,
  ]))

  result = subprocess.run(
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    args=[
      'gcloud',
      'compute',
      'instances',
      'create',
      vm_name,
      f'--project={FLAGS.project}',
      f'--zone={FLAGS.zone}',
      '--image=c1-deeplearning-common-cu100-v20201015-ubuntu-1804',
      '--image-project=ml-images',
      '--machine-type=n1-standard-96',
      '--maintenance-policy=TERMINATE',
      '--accelerator=type=nvidia-tesla-v100,count=8',
      '--scopes=cloud-platform,storage-full',
      '--boot-disk-size=256GB',
      '--boot-disk-type=pd-ssd',
      '--metadata=install-nvidia-driver=True',
      f'--metadata-from-file=startup-script={startup_file}'
    ],
  )


def main():
  pass


if __name__ == '__main__':
  app.run(main)