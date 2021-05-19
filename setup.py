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

"""setup.py for Flax."""

import os
from setuptools import find_packages
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))
try:
  README = open(os.path.join(here, "README.md"), encoding='utf-8').read()
except IOError:
  README = ""

install_requires = [
    "numpy>=1.12",
    "jax>=0.2.13",
    "matplotlib",  # only needed for tensorboard export
    "dataclasses;python_version<'3.7'", # will only install on py3.6
    "msgpack",
    "optax",
]

tests_require = [
    "atari-py",
    "clu",  # All examples.
    "gym",
    "jaxlib",
    "ml-collections",
    "opencv-python",
    "pytest",
    "pytest-cov",
    "pytest-xdist==1.34.0",  # upgrading to 2.0 broke tests, need to investigate
    "pytype",
    "sentencepiece",  # WMT example.
    "svn",
    "tensorflow-cpu>=2.4.0",
    "tensorflow_text>=2.4.0",  # WMT example.
    "tensorflow_datasets",
    "tensorflow==2.4.1",  # TODO(marcvanzee): Remove once #1326 is fixed.
]

__version__ = None

with open('flax/version.py') as f:
  exec(f.read(), globals())

setup(
    name="flax",
    version=__version__,
    description="Flax: A neural network library for JAX designed for flexibility",
    long_description="\n\n".join([README]),
    long_description_content_type='text/markdown',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    keywords="",
    author="Flax team",
    author_email="flax-dev@google.com",
    url="https://github.com/google/flax",
    packages=find_packages(),
    include_package_data=False,
    zip_safe=False,
    install_requires=install_requires,
    extras_require={
        "testing": tests_require,
        },
    )
