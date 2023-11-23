from absl.testing.absltest import TestCase

from flax.experimental import nnx


class StateTest(TestCase):
  def test_create_state(self):
    state = nnx.State({'a': nnx.Param(1), 'b': {'c': nnx.Param(2)}})

    assert state['a'] == 1
    assert state['b']['c'] == 2

  def test_get_attr(self):
    state = nnx.State({'a': nnx.Param(1), 'b': {'c': nnx.Param(2)}})

    assert state.a == 1
    assert state.b.c == 2

  def test_set_attr(self):
    state = nnx.State({'a': nnx.Param(1), 'b': {'c': nnx.Param(2)}})

    state.a = 3
    state.b.c = 4

    assert state['a'] == 3
    assert state['b']['c'] == 4

  def test_set_attr_variables(self):
    state = nnx.State({'a': nnx.Param(1), 'b': {'c': nnx.Param(2)}})

    state.a = 3
    state.b.c = 4

    assert isinstance(state.variables['a'], nnx.Param)
    assert state.variables['a'].value == 3
    assert isinstance(state.variables['b']['c'], nnx.Param)
    assert state.variables['b']['c'].value == 4
