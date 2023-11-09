import dataclasses
import threading
import typing as tp
from contextlib import contextmanager
from types import MappingProxyType


@dataclasses.dataclass
class FlagsContext(threading.local):
  flags_stack: tp.List[MappingProxyType[str, tp.Hashable]] = dataclasses.field(
    default_factory=lambda: [MappingProxyType({})]
  )


FLAGS_CONTEXT = FlagsContext()


class Flags(tp.Mapping[str, tp.Hashable]):
  __slots__ = ()

  def __getitem__(self, name: str) -> tp.Hashable:
    current_flags = FLAGS_CONTEXT.flags_stack[-1]
    if name not in current_flags:
      raise ValueError(f'Unknown Flag: {name}')
    return current_flags[name]

  __getattr__ = __getitem__

  def __iter__(self) -> tp.Iterator[str]:
    return iter(FLAGS_CONTEXT.flags_stack[-1])

  def __len__(self) -> int:
    return len(FLAGS_CONTEXT.flags_stack[-1])

  def __contains__(self, name: tp.Any) -> bool:
    return name in FLAGS_CONTEXT.flags_stack[-1]

  @contextmanager
  def __call__(self, **kwargs: tp.Hashable):
    current_flags = FLAGS_CONTEXT.flags_stack[-1]
    FLAGS_CONTEXT.flags_stack.append(
      MappingProxyType(dict(current_flags, **kwargs))
    )
    try:
      yield
    finally:
      FLAGS_CONTEXT.flags_stack.pop()

  def get(
    self, name: str, default: tp.Hashable = None
  ) -> tp.Optional[tp.Hashable]:
    return FLAGS_CONTEXT.flags_stack[-1].get(name, default)


flags = Flags()
