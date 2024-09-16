# Copyright 2024 The Flax Authors.
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

"""Sphinx directive for visualizing Flax modules.

Use directive as follows:

.. flax_module::
  :module: flax.linen
  :class: Dense
"""

import importlib

import sphinx
import sphinx.ext.autosummary.generate as ag
from docutils import nodes
from docutils.parsers.rst import directives
from docutils.statemachine import ViewList
from sphinx.util.docutils import SphinxDirective

from docs.conf_sphinx_patch import generate_autosummary_content


def render_module(modname: str, qualname: str, app):
  parent = importlib.import_module(modname)
  obj = getattr(parent, qualname)
  template = ag.AutosummaryRenderer(app)
  template_name = 'flax_module'
  imported_members = False
  recursive = False
  context = {}
  return generate_autosummary_content(
    qualname,
    obj,
    parent,
    template,
    template_name,
    imported_members,
    app,
    recursive,
    context,
    modname,
    qualname,
  )


class FlaxModuleDirective(SphinxDirective):
  has_content = True
  option_spec = {
    'module': directives.unchanged,
    'class': directives.unchanged,
  }

  def run(self):
    module_template = render_module(
      self.options['module'], self.options['class'], self.env.app
    )
    module_template = module_template.splitlines()

    # Create a container for the rendered nodes
    container_node = nodes.container()
    self.content = ViewList(module_template, self.content.parent)
    self.state.nested_parse(self.content, self.content_offset, container_node)

    return [container_node]


def setup(app):
  app.add_directive('flax_module', FlaxModuleDirective)

  return {
    'version': sphinx.__display_version__,
    'parallel_read_safe': True,
    'parallel_write_safe': True,
  }
