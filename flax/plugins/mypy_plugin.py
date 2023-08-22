from typing import Any, Callable, Optional
from mypy.plugin import Plugin, MethodSigContext, ClassDefContext
from mypy.plugins.common import add_method
from mypy.nodes import FuncDef, TypeInfo, SymbolTableNode, ArgKind
from mypy.types import AnyType, Type, FunctionLike
from mypy import types

types.TypeStrVisitor

MODULE_FULLNAME = 'flax.linen.Module'


class FlaxPlugin(Plugin):

  def get_base_class_hook(self, fullname: str):
    sym = self.lookup_fully_qualified(fullname)
    if sym and isinstance(sym.node, TypeInfo):  # pragma: no branch
      # No branching may occur if the mypy cache has not been cleared
      if any(base.fullname == MODULE_FULLNAME for base in sym.node.mro):
        return self._base_class_hook
    return None

  def _base_class_hook(self, ctx: ClassDefContext) -> None:
    ...

  def get_method_signature_hook(
      self, fullname: str
  ) -> Optional[Callable[[MethodSigContext], FunctionLike]]:
    if not fullname.endswith('__init__'):
      return None

    # print(f'get_method_signature_hook called for "{fullname}"')

    sym = self.lookup_fully_qualified(fullname)
    if sym and isinstance(sym.node, TypeInfo):  # pragma: no branch
      # No branching may occur if the mypy cache has not been cleared
      if any(base.fullname == MODULE_FULLNAME for base in sym.node.mro):
        return self.add_init_extra_args
    return None

  def add_init_extra_args(self, ctx: MethodSigContext) -> FunctionLike:
    # Check if the function is the `__init__` method
    signature = ctx.default_signature

    arg_names = signature.arg_names.copy()
    arg_kinds = signature.arg_kinds.copy()
    arg_types = signature.arg_types.copy()

    for field_name, field_type in {'name': type(str), 'parent': Any}.items():
      if field_name not in signature.arg_names:
        arg_names.append(field_name)
        arg_kinds.append(ArgKind.ARG_NAMED)
        arg_types.append(field_type)  # type: ignore

    new_sig = signature.copy_modified(
        arg_names=arg_names,
        arg_kinds=arg_kinds,
        arg_types=arg_types,
    )
    print(new_sig)
    return new_sig


def plugin(version: str):
  print('plugin called')
  return FlaxPlugin
