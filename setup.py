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

"""setup.py for Flax."""

import os
from setuptools import find_packages
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))
try:
  README = open(os.path.join(here, "README.md"), encoding="utf-8").read()
except OSError:
  README = ""

install_requires = [
    "numpy>=1.12",
    "jax>=0.4.2",
    "matplotlib",  # only needed for tensorboard export
    "msgpack",
    "optax",
    "orbax",
    "tensorstore",
    "rich>=11.1",
    "typing_extensions>=4.1.1",
    "PyYAML>=5.4.1",
]

tests_require = [
    "atari-py==0.2.5",  # Last version does not have the ROMs we test on pre-packaged
    "clu",  # All examples.
    "gym==0.18.3",
    "jaxlib",
    "jraph>=0.0.6dev0",
    "ml-collections",
    "mypy",
    "opencv-python",
    "pytest",
    "pytest-cov",
    "pytest-custom_exit_code",
    "pytest-xdist==1.34.0",  # upgrading to 2.0 broke tests, need to investigate
    "pytype",
    "sentencepiece",  # WMT/LM1B examples
    "tensorflow_text>=2.11.0",  # WMT/LM1B examples
    "tensorflow_datasets",
    "tensorflow",
    "torch",
]

__version__ = None

with open("flax/version.py") as f:
  exec(f.read(), globals())

setup(
    name="flax",
    version=__version__,
    description="Flax: A neural network library for JAX designed for flexibility",
    long_description="\n\n".join([README]),
    long_description_content_type="text/markdown",
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
    package_data={"flax": ["py.typed"]},
    zip_safe=False,
    install_requires=install_requires,
    extras_require={
        "testing": tests_require,
        },
    )
