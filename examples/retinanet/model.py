from anchor import *
from util import *
from flax.nn.initializers import normal
from jax import numpy as jnp

import flax
import jax


class ClassificationSubnet(flax.nn.Module):
  """
  This class contains the logic of the subnet which handles the classification
  at each level of the FPN. Follows: https://arxiv.org/pdf/1708.02002.pdf.
  """

  def apply(self,
            x,
            classes=1000,
            features=256,
            anchors=9,
            pi=0.01):
    """Applies the classification subnet from the RetinaNet architecture.

    Args:
      x: in model's input data, which has
      classes: the number of classes in the object detection task
      features: the number of convolution features
      anchors: the number of anchors per spatial location
      pi: a constant which is used for initializing the bias terms

    Returns:
      The output of the model, of the form (H * W * anchors, classes), where
      each row describes an anchor's probability of identifying objects
    """
    # Prepare the partial conv required in the subnet
    conv = flax.nn.Conv.partial(
        kernel_size=(3, 3), strides=(1, 1), kernel_init=normal())

    # The actual logic of the subnet
    for i in range(4):
      x = conv(x, features, name="conv_{}".format(i))
      x = flax.nn.relu(x)

    x = conv(x, classes * anchors, bias_init=pi_init(pi), name="conv_final")
    x = flax.nn.sigmoid(x)

    # After the reshape, each row will represent an anchor's classifications
    return jnp.reshape(x, (x.shape[0], -1, classes))


class RegressionSubnet(flax.nn.Module):
  """
  Contains the logic of the subnet which handles the per anchor regression at
  each level of the FPN. Follows: https://arxiv.org/pdf/1708.02002.pdf
  """

  def apply(self,
            x,
            anchor_values=4,
            anchors=9,
            features=256):
    """Applies the regression subnet from the RetinaNet architecture.

    Args:
      x: model's input data
      anchor_values: the number of values which describe an anchor
      anchors: the number of anchors per spatial location
      features: the number of convolution features

    Returns:
      The output of the model, of the form (H * W * anchors, anchor_values),
      where each row describes an anchor. As the ancor_values is usually 4, each
      row can represent the offsets of the upper left and lower right corners
      to be applied to the anchor boxes: [dx1, dy1, dx2, dy2].
    """
    # Prepare the partial modules required for the subnet
    conv = flax.nn.Conv.partial(
        kernel_size=(3, 3), strides=(1, 1), kernel_init=normal())

    # The actual logic of the subnet
    for i in range(4):
      x = conv(x, features, name="conv_{}".format(i))
      x = flax.nn.relu(x)

    x = conv(x, anchor_values * anchors, name="conv_final")

    # After the reshape, each row will represent an anchor's 4 values
    return jnp.reshape(x, (x.shape[0], -1, anchor_values))


class BottleneckBlock(flax.nn.Module):
  """
  This is an implementation of the ResNet block.
  """

  # This constant indicates the depth expansion applied on the last layer
  block_expansion = 4

  def apply(self,
            data,
            filters,
            train=True,
            downsample=False):
    """Implements the logic of a ResNet Bottleneck block.

    Args:
      data: the input of this block
      filters: the number of base filters in this block
      train: indicates whether this instance is used for training
      downsample: indicates if the input size should be downsampled. This
                  switch is also useful in determining if the feature size
                  of the input needs to be expanded, as the two phenomena
                  take place at the same time

    Returns:
      The transformed input after a Bottleneck pass
    """
    # Declare the partial modules
    final_filters = self.block_expansion * filters
    conv = flax.nn.Conv.partial(bias=False)
    batch_norm = flax.nn.BatchNorm.partial(
        use_running_average=(not train),
        momentum=0.9,
        epsilon=1e-5)

    # Process the residual such that it is compatible with the final addition
    residual = data
    mid_strides = (1, 1) if not downsample else (2, 2)
    if downsample or residual.shape[-1] != final_filters:
      residual = conv(
          data,
          final_filters, (1, 1),
          strides=mid_strides,
          name="downsample_conv")
      residual = batch_norm(residual, name="downsample_bn")

    # First 'layer'
    x = conv(data, filters, (1, 1), strides=(1, 1), name="conv_1")
    x = batch_norm(x, name="bn_1")
    x = flax.nn.relu(x)

    # Second 'layer': downsample the  HxW dimensions - like in ResNet v1.5 -
    x = conv(x, filters, (3, 3), strides=mid_strides, name="conv_2")
    x = batch_norm(x, name="bn_2")
    x = flax.nn.relu(x)

    # Third 'layer': perform addition with input
    x = conv(x, final_filters, (1, 1), strides=(1, 1), name="conv_3")
    x = batch_norm(x, name="bn_3", scale_init=flax.nn.initializers.zeros)
    x = flax.nn.relu(x + residual)

    # Return the feature map
    return x


class RetinaNet(flax.nn.Module):
  """
  This is an implementation of the RetinaNet architecture. It can be used to
  create RetinaNet instances with ResNet backbones of depth 50, 101 or 156.
  """

  # Maps to the number of blocks in the ResNet, relative to the model's depth
  depths = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 156: [3, 8, 36, 3]}

  def _nn_upsample(a, shape):
    """This method does nearest neighbor single octave upsampling on the input.

    It also features a `shape` parameter, which will trim the height and 
    width of the feature map to `shape`, if the original shape exceeds these
    dimensions.

    Args:
      a: the array getting upsampled
      shape: a list or tuple of the form [height, width, channels], indicating 
        the maximal dimensions of the output.

    Returns:
      The upsampled input, having a shape at most equal to `shape` across each 
      dimension.
    """
    upsampled = jnp.repeat(a, 2, axis=0).repeat(2, axis=1)
    return jax.lax.dynamic_slice(upsampled, [0, 0, 0],
                                 [shape[0], shape[1], shape[2]])

  nn_upsample = staticmethod(jax.vmap(_nn_upsample, in_axes=(0, None)))

  def _bottom_up_phase(self, data, train, base_features, layers):
    """Implements the backbone architecture.

    Args:
      data: the backbone's input data
      train: boolean which indicates whether the model is used for training
             or otherwise
      base_features: the number of base features in the backbone. This is
                     typically 64 according to https://arxiv.org/abs/1512.03385
      layers: a list which indicates the number of layers for each of the
              Bottleneck blocks

    Returns:
      A map of the form conv_name : str -> conv_layer_feature_map
    """
    # Collect the layer outputs such that they will be used later
    feature_maps = {}

    # C1
    x = flax.nn.Conv(
        data,
        base_features, (7, 7),
        strides=(2, 2),
        bias=False,
        name="init_conv")
    x = flax.nn.BatchNorm(
        x,
        use_running_average=(not train),
        momentum=0.9,
        epsilon=1e-5,
        name="init_bn")
    x = flax.nn.relu(x)
    x = flax.nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")

    # C2 to C5
    for block_idx in range(len(layers)):
      block = BottleneckBlock.partial(
          filters=base_features * 2**block_idx, train=train)

      start_idx = 0
      # From C2 and onward, the first block implies downsampling
      if block_idx > 0:
        start_idx = 1
        x = block(
            x, downsample=True, name="bottleneck_{}_{}".format(block_idx, 0))

      # The other blocks do not imply downsampling
      for rep in range(start_idx, layers[block_idx]):
        x = block(x, name="bottleneck_{}_{}".format(block_idx, rep))

      # Copy the output of the last layer in the conv phase for later use
      feature_maps["C{}".format(2 + block_idx)] = x

    return feature_maps

  def _top_down_phase(self, backbone_features, filters):
    """Builds the top-down phase of the RetinaNet.

    In this phase, semantically rich feature maps at different scales
    are computed from lateral connections from the backbone, and previous
    upscaled layers in the FPN.

    Args:
      backbone_features: a map of the form layer_name -> feature_map, which
                         contains the feature maps from the backbone
      filters: the number of filters in the FPN. This is generally 256,
               according to https://arxiv.org/pdf/1708.02002.pdf


    Returns:
      A map of the form layer_name -> feature_map, which contains the feature
      maps generated by the layers of the FPN
    """
    fpn_features = {}

    # Create partial for lateral connections and antialiasing operations
    conv = flax.nn.Conv.partial(features=filters, strides=(1, 1))

    # Create the supplementary feature maps P6 and P7
    fpn_features["P6"] = flax.nn.Conv(
        backbone_features["C5"],
        filters, (3, 3),
        strides=(2, 2),
        name="fpn_conv_p6")
    x = flax.nn.relu(fpn_features["P6"])
    fpn_features["P7"] = flax.nn.Conv(
        x, filters, (3, 3), strides=(2, 2), name="fpn_conv_p7")

    # Create the feature map for P5
    x = conv(
        backbone_features["C5"], kernel_size=(1, 1), name="lateral_conv_p5")
    fpn_features["P5"] = conv(
        x, kernel_size=(3, 3), name="antialiasing_conv_p5")

    # Create the feature maps P4 and P3
    for i in [4, 3]:
      lateral = conv(
          backbone_features["C{}".format(i)],
          kernel_size=(1, 1),
          name="lateral_conv_p{}".format(i))
      upsampled = self.nn_upsample(fpn_features["P{}".format(i + 1)],
                                   lateral.shape[1:])
      fpn_features["P{}".format(i)] = conv(
          upsampled + lateral,
          kernel_size=(3, 3),
          name="antialiasing_conv_p{}".format(i))

    return fpn_features

  def apply(self,
            data,
            depth=50,
            base_features=64,
            fpn_filters=256,
            anchors=9,
            anchor_values=4,
            classes=1000,
            train=True,
            k=100,
            per_class=True,
            anchors_config=None,
            img_shape=None):
    """Applies the RetinaNet architecture.

    Args:
      data: the input data. This should be of the shape (B, H, W, C)
      depth: the  depth of the ResNet backbone
      base_features: the base number of features of the backbone
      fpn_filters: the constant number of features of the FPN
      anchors: the number of anchors employed across each spatial position
      anchor_values: the number of values which characterizes an anchor
      classes: the number of classes for the object detection task
      train: indicates if the model is being used for training
      k: the `k` used in top k filtering
      per_class: True if filtering and NMS should be applied per class level
      anchors_config: an AnchorConfig object, with relevant information for
                      unpacking the anchors at various scales
      img_shape: an array of the shape (B, 3), which stores the true dimensions
        of the images in the batch (prior to padding); the rows should contain
        the [H, W, C] of each image.

    Returns:
      A dictionary of the form: feature_map_lvl : int -> (feature_map,
      classification_output, regression_output)
    """
    assert depth in self.depths, "Architecture depth is not supported."
    assert train or img_shape is not None, "Must provide image sizes for " \
                                           "clipping during training."

    layers = self.depths[depth]

    if anchors_config is None:
      anchors_config = AnchorConfig()

    # Bottom-up phase
    backbone_features = self._bottom_up_phase(data, train, base_features,
                                              layers)

    # Top-down phase
    fpn_features = self._top_down_phase(backbone_features, fpn_filters)

    # Create the partial shared models of the subnetworks
    classification_subnet = ClassificationSubnet.shared(
        classes=classes, features=fpn_filters, anchors=anchors)
    regression_subnet = RegressionSubnet.shared(
        anchor_values=anchor_values,
        features=fpn_filters,
        anchors=anchors)

    # Initialize structures relevant for training
    bboxes = jnp.zeros((data.shape[0], 0, 4))
    regressions = jnp.zeros((data.shape[0], 0, 4))
    classifications = jnp.zeros((data.shape[0], 0, classes))

    for idx, layer_idx in enumerate(range(3, 8)):
      # Get the feature maps for this subnet
      layer_input = fpn_features["P{}".format(layer_idx)]

      # Compute the regressions and the classifications, then append them
      regressions_temp = regression_subnet(layer_input)
      classifications_temp = classification_subnet(layer_input)

      regressions = jnp.append(regressions, regressions_temp, axis=1)
      classifications = jnp.append(
          classifications, classifications_temp, axis=1)

      # If not training, then expand the anchors and apply regressions
      if not train:
        # TODO: Consider moving this to its own Module
        anchors = generate_anchors(layer_input.shape[:3],
                                   anchors_config.strides[idx],
                                   anchors_config.sizes[idx],
                                   anchors_config.ratios, anchors_config.scales)
        bboxes_temp = apply_anchor_regressions(
            anchors, regressions_temp)
        bboxes_temp = clip_anchors(bboxes_temp, img_shape[:, 0], img_shape[:,
                                                                           1])
        bboxes = jnp.append(bboxes, bboxes_temp, axis=1)

    # Return the regressions, classifications, and bboxes
    return classifications, regressions, bboxes


def create_retinanet(depth, **kwargs):
  """Creates a partial RetinaNet instance.

  Args:
    depth: the depth of the ResNet backbone.
    **kwargs: additional named arguments for the `RetinaNet` partial
      instantiation. See the `RetinaNet` class definition for more details.

  Returns:
    A partially instantiated RetinaNet instance.
  """
  assert "depth" not in kwargs, "Cannot define depth twice!"

  if depth == 50:
    return RetinaNet.partial(depth=50, **kwargs)

  if depth == 101:
    return RetinaNet.partial(depth=101, **kwargs)

  if depth == 156:
    return RetinaNet.partial(depth=156, **kwargs)

  raise ValueError("The specified backbone depth is not supported!")
