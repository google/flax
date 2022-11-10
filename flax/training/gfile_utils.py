"""Gfile related helpers."""

import logging
import os
import threading

import absl.logging
from tensorflow.io import gfile  # pytype: disable=import-error


class _GFileHandler(logging.StreamHandler):
  """Writes log messages to file using gfile."""

  def __init__(self, filename, mode, flush_secs=1.0):
    super().__init__()
    gfile.makedirs(os.path.dirname(filename))
    if mode == 'a' and not gfile.exists(filename):
      mode = 'w'
    self.filehandle = gfile.GFile(filename, mode)
    self.flush_secs = flush_secs
    self.flush_timer = None

  def flush(self):
    self.filehandle.flush()

  def emit(self, record):
    msg = self.format(record)
    self.filehandle.write(f'{msg}\n')
    if self.flush_timer is not None:
      self.flush_timer.cancel()
    self.flush_timer = threading.Timer(self.flush_secs, self.flush)
    self.flush_timer.start()


def add_logger(
    workdir: str, *, basename: str = 'train', level: int = logging.INFO):
  """Starts logging to file on Google Cloud Storage bucket (GCS).

  Args:
    workdir: Directory where the logs should be stored.
    basename: Name of the log file (will have ".log" appended to it).
    level: Log level to include in the file.
  """
  path = f'{workdir.rstrip("/")}/{basename}.log'
  fh = _GFileHandler(path, 'a')
  fh.setLevel(level)
  fh.setFormatter(absl.logging.PythonFormatter())
  logging.getLogger('').addHandler(fh)
  logging.info('Started logging to "%s"', path)
