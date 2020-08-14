from dataclasses import dataclass

import jax
from jax import lax

from typing import Union, Optional, Callable, Any

@dataclass(frozen=True)
class Scan:
  axis: int

ScanAxis = Optional[int]


def scan(
    fn: Callable[..., Any],
    scan_in_axis: Any,
    scan_out_axis: Any):

  def body_fn(c, x):
    jax.tree_multimap()
    c, y = fn(c, x)
    return c, y



  def scan_fn(init, *args,
              length: Optional[int] = None, reverse: bool = False):
    
    return lax.scan(body_fn, init, args, length=length, reverse=reverse)

  return scan_fn
