
flax.cursor package
=============================

The Cursor API allows for mutability of pytrees. This API provides a more
ergonomic solution to making partial-updates of deeply nested immutable
data structures, compared to making many nested ``dataclasses.replace`` calls.

To illustrate, consider the example below::

  >>> from flax.cursor import cursor
  >>> import dataclasses
  >>> from typing import Any

  >>> @dataclasses.dataclass(frozen=True)
  >>> class A:
  ...   x: Any

  >>> a = A(A(A(A(A(A(A(0)))))))

To replace the int ``0`` using ``dataclasses.replace``, we would have to write many nested calls::

  >>> a2 = dataclasses.replace(
  ...   a,
  ...   x=dataclasses.replace(
  ...     a.x,
  ...     x=dataclasses.replace(
  ...       a.x.x,
  ...       x=dataclasses.replace(
  ...         a.x.x.x,
  ...         x=dataclasses.replace(
  ...           a.x.x.x.x,
  ...           x=dataclasses.replace(
  ...             a.x.x.x.x.x,
  ...             x=dataclasses.replace(a.x.x.x.x.x.x, x=1),
  ...           ),
  ...         ),
  ...       ),
  ...     ),
  ...   ),
  ... )

The equivalent can be achieved much more simply using the Cursor API::

  >>> a3 = cursor(a).x.x.x.x.x.x.x.set(1)
  >>> assert a2 == a3

The Cursor object keeps tracks of changes made to it and when ``.build`` is called,
generates a new object with the accumulated changes. Basic usage involves
wrapping the object in a Cursor, making changes to the Cursor object and
generating a new copy of the original object with the accumulated changes.

.. currentmodule:: flax.cursor

.. autofunction:: cursor

.. autoclass:: Cursor
  :members: apply_update, build, find, find_all, set


