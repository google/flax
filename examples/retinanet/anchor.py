from dataclasses import dataclass, field
from jax import numpy as jnp

import flax
import jax


@dataclass
class AnchorConfig:
  """This class contains the necessary information for unpacking anchors.

  It should be mentioned that the length of `sizes` and `strides` must be equal
  to the number of layers in RetinaNet's head. `scales` and `ratios` are lists
  of arbitrary lengths.
  """
  levels: list = field(default_factory=lambda: [3, 4, 5, 6, 7])
  sizes: list = field(default_factory=lambda: [32, 64, 128, 256, 512])
  strides: list = field(default_factory=lambda: [8, 16, 32, 64, 128])
  ratios: list = field(default_factory=lambda: [0.5, 1, 2])
  scales: list = field(
      default_factory=lambda: [1.0, 2.0**(1.0 / 3.0), 2.0**(2.0 / 3.0)])


def generate_base_anchors(size, ratios, scales, dtype=jnp.float32):
  """Generates candidate anchor shapes.

  Args:
    size: the size of the anchor box
    ratios: the aspect ratio of the anchor boxes
    scales: the scales of the anchor boxes
    dtype: the data type of the output

  Returns:
    The expanded anchors. The result will have the shape
    (|ratios| * |scales|, 4), where the elements on each row represent the
    upper left and lower right corners of the anchor box. The coordinates are
    relative to the origin, which is located in the center of the anchor box.
  """
  ratios = jnp.array(ratios, dtype=dtype)
  scales = jnp.array(scales, dtype=dtype)

  # Scales are replicated to enable vectorized computations
  adjusted_size = jnp.tile(scales, (2, ratios.shape[0])).T * size

  # The areas are required for extracting the correct height and width
  areas = adjusted_size[:, 0] * adjusted_size[:, 1]

  # Get the adjusted height and width, and shift the box centers in the origin
  replicated_ratios = jnp.repeat(ratios, scales.shape[0])
  heights = jnp.sqrt(areas / replicated_ratios) / 2.0
  widths = heights * replicated_ratios / 2.0
  anchors = jnp.stack((-heights, -widths, heights, widths), axis=1)

  return anchors


def generate_anchors(shape, stride, size, ratios, scales, dtype=jnp.float32):
  """Applies anchor unpacking.

  Args:
    shape: a tuple of the form (batch, height, width)
    stride: the stride of the current layer, relative to the original image
    size: the size of the window at this convolutional level
    ratios: the aspect ratios of the anchors
    scales: the scales of the anchor boxes
    dtype: the data type of the output

  Returns:
    A matrix of the shape (shape[0] * shape[1] * |ratios| * |scales|, 4).
    Each row in the returned matrix stores the coordinate of the upper left
    and lower right corner: [x1, y1, x2, y2].
  """
  # Get the anchors
  anchors = generate_base_anchors(size, ratios, scales, dtype=dtype)

  # Find the central points of the anchor boxes
  x_loc = jnp.arange(shape[2], dtype=dtype) * stride + 0.5
  y_loc = jnp.arange(shape[1], dtype=dtype) * stride + 0.5
  xx, yy = jnp.meshgrid(x_loc, y_loc)
  xx, yy = jnp.reshape(xx, -1), jnp.reshape(yy, -1)

  # Apply all anchor boxes to every spatial location in the image
  centers = jnp.stack((xx, yy), axis=1)
  centers = jnp.repeat(jnp.tile(centers, (1, 2)), anchors.shape[0], axis=0)
  centers = centers + jnp.tile(anchors, (xx.shape[0], 1))

  # Tile the map for each image in the batch
  return jnp.tile(jnp.expand_dims(centers, axis=0), (shape[0], 1, 1))


def clip_anchors(anchors, shape):
  """Clips the anchors, such that they do not exceed `shape`

  Args:
    anchors: an (|A|, 4) matrix, where |A| is the number of anchors; the 
      4 elements on each row represent the anchor coordinates
    shape: a list or tuple of 2 elements: (height, width)

  Returns:
    A matrix of the form (|A|, 4), which contains the clipped anchors
  """
  anchors = jax.ops.index_update(
      anchors, jax.ops.index[:, 0],
      jnp.minimum(jnp.maximum(anchors[:, 0], 0), shape[1]))
  anchors = jax.ops.index_update(
      anchors, jax.ops.index[:, 1],
      jnp.minimum(jnp.maximum(anchors[:, 1], 0), shape[0]))
  anchors = jax.ops.index_update(
      anchors, jax.ops.index[:, 2],
      jnp.minimum(jnp.maximum(anchors[:, 2], 0), shape[1]))
  anchors = jax.ops.index_update(
      anchors, jax.ops.index[:, 3],
      jnp.minimum(jnp.maximum(anchors[:, 3], 0), shape[0]))
  return anchors


def generate_all_anchors(shape,
                         levels,
                         strides,
                         sizes,
                         ratios,
                         scales,
                         clip=False,
                         dtype=jnp.float32):
  """Generate all the anchors for an image of a given size.

  More specifically, given an image size, this method generates the entire 
  set of candidate anchors at various scales for for the image size.

  Args:
    shape: a tuple of the form (height, width)
    levels: a list indicating the subsampling levels
    strides: a list of strides corresponding to each subsampling level
    sizes: a list of sizes in the original image of the anchors at each level
    ratios: the aspect ratios for the anchors
    scales: the anchor scales
    clip: True if the anchor coordinates should be clipped to not exceed the 
      bounds imposed by the `shape` parameter
    dtype: the data type of the output
    
  Returns:
    A matrix of the shape (-1, 5), where -1 is replaced by the total number of 
    generated anchors. The first 4 columns indicate [x1, y1, x2, y2] of each
    anchor, whereas the last column can be used to indicate the status of 
    the anchor: background, ignored, foreground.
  """
  assert len(levels) == len(strides) == len(sizes), "levels, strides and sizes"\
                                                    "must have the same length"

  # Compute the feature map sizes
  shape = jnp.array(shape)
  feature_maps = [(shape + 2**level - 1) // (2**level) for level in levels]

  # Stack the anchors on axis 0: first levels[0], ..., levels[-1]
  anchors = jnp.zeros((1, 0, 4), dtype=dtype)
  for idx, feature_map in enumerate(feature_maps):
    res = generate_anchors((1,) + tuple(feature_map), strides[idx], sizes[idx],
                           ratios, scales, dtype)
    anchors = jnp.append(anchors, res, axis=1)

  anchors = anchors[0, :, :]

  # Clip the anchors such that their coords do not fall outside the image bounds
  if clip:
    anchors = clip_anchors(anchors, shape)

  extra_zeros = jnp.zeros((anchors.shape[0], 1), dtype=dtype)
  return jnp.concatenate((anchors, extra_zeros), axis=1)


def apply_anchor_regressions(anchors,
                             regressions,
                             means=None,
                             devs=None,
                             dtype=jnp.float32):
  """Applies the regression values to the raw anchor boxes.

  Args:
    anchors: a matrix which holds the anchor information
    regressions: the regression values generated by the
    means: a list of means across each of the regression dimensions
    devs: a list of standard deviations across each of the regression
              dimensions
    dtype: the data type of the output

  Returns:
    The adjusted anchor boxes, relative to the regressions. The shape will be
    the same as the `anchor` input.
  """
  if means is None:
    means = jnp.array([0] * anchors.shape[1], dtype=dtype)
  elif isinstance(means, (list, tuple)):
    means = jnp.array(means, dtype=dtype)

  if devs is None:
    devs = jnp.array([0.2] * anchors.shape[1], dtype=dtype)
  elif isinstance(devs, (list, tuple)):
    devs = jnp.array(devs, dtype=dtype)

  width = anchors[:, :, 2] - anchors[:, :, 0]
  height = anchors[:, :, 3] - anchors[:, :, 1]

  x1 = anchors[:, :, 0] + (regressions[:, :, 0] * devs[0] + means[0]) * width
  y1 = anchors[:, :, 1] + (regressions[:, :, 1] * devs[1] + means[1]) * height
  x2 = anchors[:, :, 2] + (regressions[:, :, 2] * devs[2] + means[2]) * width
  y2 = anchors[:, :, 3] + (regressions[:, :, 3] * devs[3] + means[3]) * height

  return jnp.stack((x1, y1, x2, y2), axis=-1)
