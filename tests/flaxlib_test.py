from absl.testing import absltest
import flaxlib


class TestFlaxlib(absltest.TestCase):
  def test_flaxlib(self):
    self.assertEqual(flaxlib.sum_as_string(1, 2), '3')
