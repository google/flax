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

# patch sphinx
import docs.conf_sphinx_patch
# -- Project information -----------------------------------------------------

project = 'Flax'
copyright = '2023, The Flax authors'  # pylint: disable=redefined-builtin
author = 'The Flax authors'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'myst_nb',
    'codediff',
    'sphinx_design',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The suffix(es) of source filenames.
# Note: important to list ipynb before md here: we have both md and ipynb
# copies of each notebook, and myst will choose which to convert based on
# the order in the source_suffix list. Notebooks which are not executed have
# outputs stored in ipynb but not in md, so we must convert the ipynb.
source_suffix = ['.rst', '.ipynb', '.md']

autosummary_generate = True

master_doc = 'index'

autodoc_typehints = 'none'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'pydata_sphinx_theme'
html_theme = 'sphinx_book_theme'
html_css_files = ['css/flax_theme.css']

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = './flax.png'
html_favicon = './flax.png'

# title of the website
html_title = ''

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named 'default.css' will overwrite the builtin 'default.css'.
html_static_path = ['_static']

html_theme_options = {
    'repository_url': 'https://github.com/google/flax',
    'use_repository_button': True,     # add a 'link to repository' button
    'use_issues_button': False,        # add an 'Open an Issue' button
    'path_to_docs': 'docs',            # used to compute the path to launch notebooks in colab
    'launch_buttons': {
        'colab_url': 'https://colab.research.google.com/',
    },
    'prev_next_buttons_location': None,
    'show_navbar_depth': 1,
}

# -- Options for myst ----------------------------------------------
# uncomment line below to avoid running notebooks during development
# nb_execution_mode = 'off'
# Notebook cell execution timeout; defaults to 30.
nb_execution_timeout = 100
# List of patterns, relative to source directory, that match notebook
# files that will not be executed.
myst_enable_extensions = ['dollarmath']
nb_execution_excludepatterns = [
  'getting_started.ipynb', # <-- times out
  'optax_update_guide.ipynb', # <-- requires flax<=0.5.3
]
# raise exceptions on execution so CI can catch errors
nb_execution_allow_errors = False
nb_execution_raise_on_error = True

# -- Extension configuration -------------------------------------------------

# Tell sphinx-autodoc-typehints to generate stub parameter annotations including
# types, even if the parameters aren't explicitly documented.
always_document_param_types = True
