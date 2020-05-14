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

version = "0.1.0"

here = os.path.abspath(os.path.dirname(__file__))
try:
  README = open(os.path.join(here, "README.md"), encoding='utf-8').read()
except IOError:
  README = ""

install_requires = [
    "numpy>=1.12",
    "jax>=0.1.59",
    "matplotlib",  # only needed for tensorboard export
    "dataclasses",  # will only install on py3.6
    "msgpack",
]

tests_require = [
    "jaxlib",
    "pytest",
    "pytest-cov",
    "pytest-xdist",
    "tensorflow",
    "tensorflow_datasets",
]

setup(
    name="flax",
    version=version,
    description="Flax: A neural network library for JAX designed for flexibility",
    long_description="\n\n".join([README]),
    long_description_content_type='text/markdown',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    keywords="",
    author="Flax team",
    author_email="flax-dev@google.com",
    url="https://github.com/google/flax",
    license="Apache",
    packages=find_packages(),
    include_package_data=False,
    zip_safe=False,
    install_requires=install_requires,
    extras_require={
        "testing": tests_require,
        },
    )
