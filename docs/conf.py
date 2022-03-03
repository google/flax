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

"""Configuration file for the Sphinx documentation builder."""


# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import os
import sys
sys.path.insert(0, os.path.abspath('..'))
# Include local extension.
sys.path.append(os.path.abspath('./_ext'))

# -- Project information -----------------------------------------------------

project = 'Flax'
copyright = '2020, The Flax authors'  # pylint: disable=redefined-builtin
author = 'The Flax authors'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'nbsphinx',
    'recommonmark',
    'codediff',
    'sphinx_markdown_tables'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

source_suffix = ['.rst', '.md']

autosummary_generate = True

master_doc = 'index'

autodoc_typehints = 'description'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_style = 'css/flax_theme.css'

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = './flax.png'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

nbsphinx_codecell_lexer = 'ipython3'

nbsphinx_prolog = r"""
{% set docname = 'docs/' + env.doc2path(env.docname, base=None) %}

.. only:: html

    .. role:: raw-html(raw)
        :format: html

    .. nbinfo::

        :raw-html:`<a href="https://colab.research.google.com/github/google/flax/blob/main/{{ docname }}"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg" style="vertical-align:text-bottom"></a>`
        :raw-html:`<a href="https://github.com/google/flax/blob/main/{{ docname }}"><img alt="Open On GitHub" src="https://img.shields.io/badge/Open-on%20GitHub-blue?logo=GitHub" style="vertical-align:text-bottom"></a>`


"""

# -- Extension configuration -------------------------------------------------

# Tell sphinx-autodoc-typehints to generate stub parameter annotations including
# types, even if the parameters aren't explicitly documented.
always_document_param_types = True
