"""Hooks are actions that are executed during the training loop."""

import abc
from typing import Optional

from absl import logging


class Hook(abc.ABC):
  """Interface for all hooks."""

  @abc.abstractmethod
  def __call__(self, step: int, t: float):
    pass


class EveryNHook(Hook):
  """Abstract base class for hooks that are executed periodically."""

  def __init__(self,
               *,
               every_steps: Optional[int] = None,
               every_secs: Optional[float] = None):
    self._every_steps = every_steps
    self._every_secs = every_secs
    self._previous_step = None
    self._previous_time = None
    self._last_step = None

  def _apply_condition(self, step: int, t: float):
    if self._every_steps is not None and step % self._every_steps == 0:
      return True
    if (self._every_secs is not None and
        t - self._previous_time > self._every_secs):
      return True
    return False

  def __call__(self, step: int, t: float):
    """Method to call the hook after every training step."""
    if self._previous_step is None:
      self._previous_step = step
      self._previous_time = t
      self._last_step = step
      return

    if self._every_steps is not None:
      if step - self._last_step != 1:
        raise ValueError("EveryNHook must be called after every step (once).")
    self._last_step = step

    if self._apply_condition(step, t):
      self._apply(step, t)
      self._previous_step = step
      self._previous_time = t

  @abc.abstractmethod
  def _apply(self, step: int, t: float):
    pass


class ReportProgress(EveryNHook):
  """This hook will set the progress note on the work unit."""

  def __init__(self,
               *,
               num_train_steps: int,
               every_steps: Optional[int] = None,
               every_secs: Optional[float] = 60.0):
    """Creates a new ReportProgress hook.

    Warning: The progress and the reported steps_per_sec are estimates. We
    ignore the asynchronous dispatch for JAX and other operations in the
    training loop (e.g. evaluation).

    Args:
      num_train_steps: The total number of training steps for training.
      every_steps: How often to report the progress in number of training steps.
      every_secs: How often to report progress as time interval.
    """
    super().__init__(every_steps=every_steps, every_secs=every_secs)
    self._num_train_steps = num_train_steps

  def _apply_condition(self, step: int, t: float):
    # Always trigger at last step.
    if step == self._num_train_steps:
      return True
    return super()._apply_condition(step, t)

  def _apply(self, step: int, t: float):
    steps_per_sec = (step - self._previous_step) / (t - self._previous_time)
    eta_seconds = (self._num_train_steps - step) / steps_per_sec
    message = (f"{100 * step / self._num_train_steps:.1f}% @{step}, "
               f"{steps_per_sec:.1f} steps/s, ETA: {eta_seconds / 60:.0f} min")
    logging.info(message)
