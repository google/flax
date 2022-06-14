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

"""Patch Sphinx to improve documentation aesthetics."""

# TODO(cgarciae): Send a PR to sphinx to upstream this fix. Issue: https://github.com/google/flax/issues/2196
# This patch is needed to make autosummary provide the "annotations"
# variable so we can exclude function attributes from the methods list
# in flax_module.rst. The patch as such only adds this single line:
#
#     ns['annotations'] = list(getattr(obj, '__annotations__', {}).keys())'
#
# We should consider sending a PR to sphinx so we can get rid of this.
# Original source: https://github.com/sphinx-doc/sphinx/blob/0aedcc9a916daa92d477226da67d33ce1831822e/sphinx/ext/autosummary/generate.py#L211-L351
from typing import Any, Dict, List, Set, Tuple
import sphinx.ext.autosummary.generate as ag
import sphinx.ext.autodoc

def generate_autosummary_content(name: str, obj: Any, parent: Any,
                                 template: ag.AutosummaryRenderer, template_name: str,
                                 imported_members: bool, app: Any,
                                 recursive: bool, context: Dict,
                                 modname: str = None, qualname: str = None) -> str:
    doc = ag.get_documenter(app, obj, parent)

    def skip_member(obj: Any, name: str, objtype: str) -> bool:
        try:
            return app.emit_firstresult('autodoc-skip-member', objtype, name,
                                        obj, False, {})
        except Exception as exc:
            ag.logger.warning(__('autosummary: failed to determine %r to be documented, '
                              'the following exception was raised:\n%s'),
                           name, exc, type='autosummary')
            return False

    def get_class_members(obj: Any) -> Dict[str, Any]:
        members = sphinx.ext.autodoc.get_class_members(obj, [qualname], ag.safe_getattr)
        return {name: member.object for name, member in members.items()}

    def get_module_members(obj: Any) -> Dict[str, Any]:
        members = {}
        for name in ag.members_of(obj, app.config):
            try:
                members[name] = ag.safe_getattr(obj, name)
            except AttributeError:
                continue
        return members

    def get_all_members(obj: Any) -> Dict[str, Any]:
        if doc.objtype == "module":
            return get_module_members(obj)
        elif doc.objtype == "class":
            return get_class_members(obj)
        return {}

    def get_members(obj: Any, types: Set[str], include_public: List[str] = [],
                    imported: bool = True) -> Tuple[List[str], List[str]]:
        items: List[str] = []
        public: List[str] = []

        all_members = get_all_members(obj)
        for name, value in all_members.items():
            documenter = ag.get_documenter(app, value, obj)
            if documenter.objtype in types:
                # skip imported members if expected
                if imported or getattr(value, '__module__', None) == obj.__name__:
                    skipped = skip_member(value, name, documenter.objtype)
                    if skipped is True:
                        pass
                    elif skipped is False:
                        # show the member forcedly
                        items.append(name)
                        public.append(name)
                    else:
                        items.append(name)
                        if name in include_public or not name.startswith('_'):
                            # considers member as public
                            public.append(name)
        return public, items

    def get_module_attrs(members: Any) -> Tuple[List[str], List[str]]:
        """Find module attributes with docstrings."""
        attrs, public = [], []
        try:
            analyzer = ag.ModuleAnalyzer.for_module(name)
            attr_docs = analyzer.find_attr_docs()
            for namespace, attr_name in attr_docs:
                if namespace == '' and attr_name in members:
                    attrs.append(attr_name)
                    if not attr_name.startswith('_'):
                        public.append(attr_name)
        except ag.PycodeError:
            pass    # give up if ModuleAnalyzer fails to parse code
        return public, attrs

    def get_modules(obj: Any) -> Tuple[List[str], List[str]]:
        items: List[str] = []
        for _, modname, _ispkg in ag.pkgutil.iter_modules(obj.__path__):
            fullname = name + '.' + modname
            try:
                module = ag.import_module(fullname)
                if module and hasattr(module, '__sphinx_mock__'):
                    continue
            except ImportError:
                pass

            items.append(fullname)
        public = [x for x in items if not x.split('.')[-1].startswith('_')]
        return public, items

    ns: Dict[str, Any] = {}
    ns.update(context)

    if doc.objtype == 'module':
        scanner = ag.ModuleScanner(app, obj)
        ns['members'] = scanner.scan(imported_members)
        ns['functions'], ns['all_functions'] = \
            get_members(obj, {'function'}, imported=imported_members)
        ns['classes'], ns['all_classes'] = \
            get_members(obj, {'class'}, imported=imported_members)
        ns['exceptions'], ns['all_exceptions'] = \
            get_members(obj, {'exception'}, imported=imported_members)
        ns['attributes'], ns['all_attributes'] = \
            get_module_attrs(ns['members'])
        ispackage = hasattr(obj, '__path__')
        if ispackage and recursive:
            ns['modules'], ns['all_modules'] = get_modules(obj)
    elif doc.objtype == 'class':
        ns['members'] = dir(obj)
        ns['inherited_members'] = \
            set(dir(obj)) - set(obj.__dict__.keys())
        ns['methods'], ns['all_methods'] = \
            get_members(obj, {'method'}, ['__init__'])
        ns['attributes'], ns['all_attributes'] = \
            get_members(obj, {'attribute', 'property'})
        ns['annotations'] = list(getattr(obj, '__annotations__', {}).keys())

    if modname is None or qualname is None:
        modname, qualname = ag.split_full_qualified_name(name)

    if doc.objtype in ('method', 'attribute', 'property'):
        ns['class'] = qualname.rsplit(".", 1)[0]

    if doc.objtype in ('class',):
        shortname = qualname
    else:
        shortname = qualname.rsplit(".", 1)[-1]

    ns['fullname'] = name
    ns['module'] = modname
    ns['objname'] = qualname
    ns['name'] = shortname

    ns['objtype'] = doc.objtype
    ns['underline'] = len(name) * '='

    if template_name:
        return template.render(template_name, ns)
    else:
        return template.render(doc.objtype, ns)

ag.generate_autosummary_content = generate_autosummary_content
