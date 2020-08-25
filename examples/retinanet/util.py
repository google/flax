from jax import numpy as jnp

import jax
import numpy as np
import tensorflow as tf

CATEGORY_MAP = {
    0: "background",
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    12: "stop sign",
    13: "parking meter",
    14: "bench",
    15: "bird",
    16: "cat",
    17: "dog",
    18: "horse",
    19: "sheep",
    20: "cow",
    21: "elephant",
    22: "bear",
    23: "zebra",
    24: "giraffe",
    25: "backpack",
    26: "umbrella",
    27: "handbag",
    28: "tie",
    29: "suitcase",
    30: "frisbee",
    31: "skis",
    32: "snowboard",
    33: "sports ball",
    34: "kite",
    35: "baseball bat",
    36: "baseball glove",
    37: "skateboard",
    38: "surfboard",
    39: "tennis racket",
    40: "bottle",
    41: "wine glass",
    42: "cup",
    43: "fork",
    44: "knife",
    45: "spoon",
    46: "bowl",
    47: "banana",
    48: "apple",
    49: "sandwich",
    50: "orange",
    51: "broccoli",
    52: "carrot",
    53: "hot dog",
    54: "pizza",
    55: "donut",
    56: "cake",
    57: "chair",
    58: "couch",
    59: "potted plant",
    60: "bed",
    61: "dining table",
    62: "toilet",
    63: "tv",
    64: "laptop",
    65: "mouse",
    66: "remote",
    67: "keyboard",
    68: "cell phone",
    69: "microwave",
    70: "oven",
    71: "toaster",
    72: "sink",
    73: "refrigerator",
    74: "book",
    75: "clock",
    76: "vase",
    77: "scissors",
    78: "teddy bear",
    79: "hair drier",
    80: "toothbrush"
}


def pi_init(pi):
  """Wrapper to log-based weight initializer function.

  This initializer is used for the bias term in the classification subnet, as
  described in https://arxiv.org/abs/1708.02002

  Args:
    pi: the prior probability of detecting an object

  Returns:
    An array for initializing a module's weights / biases
  """

  def _inner(key, shape, dtype=jnp.float32):
    return jnp.ones(shape, dtype) * (-jnp.log((1 - pi) / pi))

  return _inner


@jax.vmap
def clip_anchors(anchors, height, width):
  """Clips anchors to height and width of image.

  More specifically, the x coordinates of the base anchors are clipped,
  such that they are always found in the `[0, width]` interval, and
  the `y` coordinates are always found in the `[0, height]` interval.

  Args:
    anchors: a tensor of the shape (|A|, 4) where each row contains 
      the `[x1, y1, x2, y1]` of that anchor
    height: the height of the image
    width: the width of the image

  Returns:
    A matrix of the form (|A|, 4), which contains the clipped anchors, as well
    as an extra column which can be used to store the status of the anchor.
  """
  x1 = jnp.clip(anchors[:, 0], 0.0, width)
  y1 = jnp.clip(anchors[:, 1], 0.0, height)
  x2 = jnp.clip(anchors[:, 2], 0.0, width)
  y2 = jnp.clip(anchors[:, 3], 0.0, height)
  return jnp.stack([x1, y1, x2, y2], axis=1)


def non_max_suppression(bboxes, scores, t):
  """Implements the Non-Maximum Suppression algorithm.

  More specifically, this algorithm retains the bboxes based on their scores 
  (those that have a higher score are favored), and IoU's with the other bboxes
  (bboxes that have a high overlap with bboxes with higher scores are removed).

  Args:
    bboxes: a matrix of the form (|B|, 4), where |B| is the number of bboxes,
      and the columns represent the coordinates of each bbox: [x1, y1, x2, y2]
    scores: a vector of the form (|B|,) storing the confidence in each bbox
    t: the IoU threshold; overlap above this threshold with higher scoring 
      bboxes will imply the lower scoring bbox should be discarded

  Returns:
    The indexes of the bboxes which are retained after NMS is applied, as well
    as their indexes in the original matrix.
  """
  selected_idx = []

  # Split the bboxes so they're easier to manipulate throughout
  x1 = bboxes[:, 0]
  y1 = bboxes[:, 1]
  x2 = bboxes[:, 2]
  y2 = bboxes[:, 3]

  sorted_idx = jnp.argsort(scores)
  areas = (x2 - x1 + 1) * (y2 - y1 + 1)

  while sorted_idx.shape[0] > 0:
    # Select the index of the bbox with the highest score
    current = sorted_idx[-1]
    selected_idx.append(current)

    # Determine the height and width of the intersections with the current bbox
    xx1 = jnp.maximum(x1[current], x1[sorted_idx[:-1]])
    yy1 = jnp.maximum(y1[current], y1[sorted_idx[:-1]])
    xx2 = jnp.minimum(x2[current], x2[sorted_idx[:-1]])
    yy2 = jnp.minimum(y2[current], y2[sorted_idx[:-1]])

    width = jnp.maximum(0.0, xx2 - xx1 + 1)
    height = jnp.maximum(0.0, yy2 - yy1 + 1)

    # Compute the IoU between the current bbox and all the other bboxes
    intersection = width * height
    ious = intersection / (
        areas[current] + areas[sorted_idx[:-1]] - intersection)

    # Keep only the bboxes with the lower threshold
    sorted_idx = sorted_idx[jnp.where(ious < t)[0]]

  # Return the indexes of the non-suppressed bboxes
  selected_idx = jnp.array(selected_idx, dtype=jnp.int32)
  return bboxes[selected_idx, :], selected_idx


def vertical_pad(data, pad_count, dtype=jnp.float32):
  """Applies vertical padding to the data by adding extra rows with 0.
  
  Args:
    data: the data to be padded 
    pad_count: the number of extra rows of padding to be added to the data

  Returns:
    `data` with extra padding
  """
  pad_shape = (pad_count,) + data.shape[1:]
  pad_structure = jnp.zeros(pad_shape, dtype=dtype)
  return jnp.append(data, pad_structure, axis=0)


def _top_k(scores, k, t=0.0):
  """Applies top k selection on the `scores` parameter.

  Args:
    scores: a vector of arbitrary length, containing non-negative scores, from 
      which only the at most top `k` highest scoring entries are selected.
    k: the maximal number of elements to be selected from `scores`
    t: a thresholding parameter (inclusive) which is applied on `scores`; 
      elements failing to meet the threshold are removed 

  Returns:
    Top top k entries from `scores` after thresholding with `t` is applied,
    as well as their indexes in the original vector.
  """
  scores = jnp.where(scores >= t, scores, 0.0)
  idx = jnp.argsort(scores)[-k:]
  return scores[idx], idx


top_k = jax.vmap(_top_k, in_axes=(0, None, None))


def filter_by_score(bboxes,
                    scores,
                    score_threshold: float = 0.05,
                    k: int = 1000,
                    per_class=False):
  """Apply top-k filtering on the bbox scores.

  More specifically, apply top `k` selection on the bboxes by filtering 
  out the bboxes which have a score lower than the `score_threshold`. The 
  filtering can be done either at the per class level or across all classes
  at the same time, as indicated by `per_class`.

  Args:
    bboxes: a matrix of the shape (|B|, 4), where |B| is the number of bboxes;
      each row will store the bbox's 4 coordinates: [x1, y1, x2, y2]   
    scores: a matrix of the shape (|B|, K), where K is the number of classes;
      each entry in a row will store the classification confidence of that class
    score_threshold: bboxes having a confidence score lower than this parameter
      will be discarded.
    k: the `k` parameter of the top k selection
    per_class: a flag which indicates whether the filtering and NMS operations 
      should be executed on a per class level    

  Returns:
    The bboxes retained after the filtering, and the indexes of these bboxes
    in the original data structure.
  """

  def _filter(inner_scores, labels):
    # First apply the top k filtering on the input
    top_k_scores, top_k_idx = top_k(inner_scores, k, score_threshold)
    top_k_bboxes = bboxes[top_k_idx, :]
    top_k_labels = labels[top_k_idx]
    return top_k_bboxes, top_k_scores, top_k_labels

  # Apply per class filtering
  if per_class:
    row_count = scores.shape[0]

    # Create some accumulators
    bbox_acc = jnp.zeros((0, 4))
    scores_acc = jnp.zeros(0)
    label_acc = jnp.zeros(0)

    # Iterate through all the classes, and apply filtering
    for i in range(scores.shape[1]):
      current_labels = jnp.ones(row_count, dtype=jnp.int32) * i
      current_scores = scores[:, i]
      temp_bboxes, temp_scores, temp_labels = _filter(current_scores,
                                                      current_labels)
      bbox_acc = jnp.append(bbox_acc, temp_bboxes, axis=0)
      scores_acc = jnp.append(scores_acc, temp_scores, axis=0)
      label_acc = jnp.append(label_acc, temp_labels, axis=0)

    return bbox_acc, scores_acc, label_acc
  else:
    current_labels = jnp.argmax(scores, axis=1)
    current_scores = scores[jnp.arange(scores.shape[0]), current_labels]
    return _filter(current_scores, current_labels)


def batch_filter_by_score(bboxes,
                          scores,
                          max_rows,
                          score_threshold: float = 0.05,
                          k: int = 1000,
                          per_class=False):
  """Apply top-k filtering on a batch. 

  For further explanation see `filter_by_score` documentation.

  Args:
    bboxes: an array of the shape (N, |B|, 4), where N is the batch count and 
      |B| is the number of bboxes; each row will store the bbox's 4 
      coordinates: [x1, y1, x2, y2]   
    scores: a matrix of the shape (N, |B|, K), where K is the number of classes;
      each entry in a row will store the classification confidence of that class
    max_rows: a scalar which indicates the number of output rows for each data
      structure, obtained either through padding or trimming 
    score_threshold: bboxes having a confidence score lower than this parameter
      will be discarded.
    k: the `k` parameter of the top k selection. This value will also be used
      to pad the outputs if the number of valid bboxes is lower than `k`
    per_class: a flag which indicates whether the filtering and NMS operations 
      should be executed on a per class level; if True, then the output will 
      be padded to the size of `k x scores.shape[-1]`   

  Returns:
    A tuple containing: the number of true bboxes (i.e. not obtained through
    padding), the padded bboxes, the padded scores, the padded labels.  
  """

  def _inner(idx):
    # Isolate the relevant image data and apply the filtering
    temp_bboxes = bboxes[idx, ...]
    temp_scores = scores[idx, ...]
    temp_bbox, temp_scores, temp_labels = filter_by_score(
        temp_bboxes, temp_scores, score_threshold, k, per_class)

    # Pad or trim if necessary
    count = temp_bbox.shape[0]
    if count < max_rows:
      delta = max_rows - count
      temp_bbox = vertical_pad(temp_bbox, delta)
      temp_scores = vertical_pad(temp_scores, delta)
      temp_labels = vertical_pad(temp_labels, delta)
    elif count > max_rows:
      # Skip the first rows, as those have lower confidence
      temp_bbox = temp_bbox[-max_rows:, ...]
      temp_scores = temp_scores[-max_rows:]
      temp_labels = temp_labels[-max_rows:]

    return count, temp_bbox, temp_scores, temp_labels

  # Create the structures which will store the filtered bboxes
  batch_size = bboxes.shape[0]
  counts = np.zeros(batch_size, dtype=int)
  filtered_bboxes = np.zeros((batch_size, max_rows, 4))
  filtered_scores = np.zeros((batch_size, max_rows))
  filtered_labels = np.zeros((batch_size, max_rows), dtype=int)

  # Apply the filtering function on each batch entry
  # FIXME: Find a way to do this in parallel
  for idx, pack in enumerate(map(_inner, np.arange(batch_size))):
    counts[idx], filtered_bboxes[idx], filtered_scores[idx], filtered_labels[
        idx] = pack

  # Convert to jnp and return
  counts = jnp.array(counts, dtype=jnp.int32)
  filtered_bboxes = jnp.array(filtered_bboxes)
  filtered_scores = jnp.array(filtered_scores)
  filtered_labels = jnp.array(filtered_labels, dtype=jnp.int32)
  return counts, filtered_bboxes, filtered_scores, filtered_labels


def generate_inferences(bboxes,
                        scores,
                        labels,
                        counts,
                        nms=True,
                        classes=-1,
                        per_class=False,
                        iou_threshold=0.5,
                        max_outputs=100):
  """Generates the batch bbox predictions.

  More specifically, given a dictionary which stores the filtered outputs 
  of the regression and classification subnets, this function generates the 
  final predictions for the batch, by concatenating the valid parts of 
  the outputs together. This method can optionally also apply Non-Maximum 
  Suppression (NMS) on the input. NMS can be applied at a per-class level 
  or on all the classes at the same time.  

  Args:
    bboxes: an array having the dimensions (N, L, A, 4), where `N` is the batch
      size, `L` is the number of levels, and `A` is the number of (padded) 
      anchors; this stores the 4 anchor coordinates.
    scores: an array having the shape (N, L, A), which stores the score 
      associated with each anchor
    labels: an array having the shape (N, L, A), which stores the label 
      associated with each anchor
    counts: an array of the shape (N, L), which stores the number of true
      anchors; this number is useful towards discarding the padded anchors
    nms: if True, enables the use of NMS
    classes: the number of classes in the classification task; if `per_class`
      is True, this argument must also be specified
    per_class: if True enables NMS on a per-class level
    max_outputs: a scalar which indicates the maximal number of predicted bboxes
      generated by this method. If there are fewer than predictions, then the
      outputs will be padded. 

  Returns:
    A dictionary of the following form:

    ```
    {
      "counts": <count_list>,  # Shape: (<batch_size>)
      "bboxes": <bboxes>,  # Shape: (<batch_size, max_outputs, 4>)
      "scores": <scores>,  # Shape: (<batch_size, max_outputs>)
      "labels": <labels>   # Shape: (<batch_size, max_outputs>)
    }
    ```

    Here, `counts` holds the number of usable elements, as the output will be 
    automatically padded or trimmed to `max_outputs`
  """
  assert not per_class or per_class and classes > 0, "If per_class is True, " \
    "then classes must be a positive integer"

  def _inner(idx):
    # Isolate the relevant image data
    t_bboxes = bboxes[idx, ...]
    t_scores = scores[idx, ...]
    t_labels = labels[idx, ...]
    t_counts = counts[idx]

    # Prepare the accumulators
    acc_bboxes = jnp.zeros((0, 4))
    acc_scores = jnp.zeros(0)
    acc_labels = jnp.zeros(0, dtype=jnp.int32)

    # Concatenate the image data into one
    levels = t_bboxes.shape[0]
    for i in range(levels):
      count = t_counts[i]
      acc_bboxes = jnp.append(acc_bboxes, t_bboxes[i, :count, :], axis=0)
      acc_scores = jnp.append(acc_scores, t_scores[i, :count], axis=0)
      acc_labels = jnp.append(acc_labels, t_labels[i, :count], axis=0)

    # Apply NMS if requested
    if nms:
      if per_class:
        # Create new acc datastructures
        t_acc_bboxes = jnp.zeros((0, 4))
        t_acc_scores = jnp.zeros(0)
        t_acc_labels = jnp.zeros(0, dtype=jnp.int32)

        # Apply NMS for each label
        for i in range(classes):
          # Isolate the label data
          idx = jnp.where(acc_labels == i)[0]
          t_bboxes = acc_bboxes[idx]
          t_scores = acc_scores[idx]
          t_labels = acc_labels[idx]

          # Apply NMS and isolate the data of the top bboxes
          t_bboxes, nms_idx = non_max_suppression(t_bboxes, t_scores,
                                                  iou_threshold)
          t_scores = t_scores[nms_idx]
          t_labels = t_labels[nms_idx]

          # Append the filtered bboxes
          t_acc_bboxes = jnp.append(t_acc_bboxes, t_bboxes, axis=0)
          t_acc_scores = jnp.append(t_acc_scores, t_scores, axis=0)
          t_acc_labels = jnp.append(t_acc_labels, t_labels, axis=0)

        # Apply top-k selection based on confidence
        acc_scores, idx = top_k(t_acc_scores, max_outputs)
        acc_bboxes = t_acc_bboxes[idx]
        acc_labels = t_acc_labels[idx]
      else:
        acc_bboxes, nms_idx = non_max_suppression(acc_bboxes, acc_scores,
                                                  iou_threshold)
        acc_scores = acc_scores[nms_idx]
        acc_labels = acc_labels[nms_idx]

    # Pad or trim the outputs such that they have `max_outputs` rows
    count = acc_bboxes.shape[0]
    if count < max_outputs:
      delta = max_outputs - count
      acc_bboxes = vertical_pad(acc_bboxes, delta)
      acc_scores = vertical_pad(acc_scores, delta)
      acc_labels = vertical_pad(acc_labels, delta)
    elif count > max_outputs:
      acc_bboxes = acc_bboxes[:max_outputs, ...]
      acc_scores = acc_scores[:max_outputs]
      acc_labels = acc_labels[:max_outputs]

    # Return the results of this stage
    return count, acc_bboxes, acc_scores, acc_labels

  # Determine the batch size and initialize accumulator datastructures
  batch_size = bboxes.shape[0]
  f_counts = np.zeros(batch_size, dtype=int)
  f_bboxes = np.zeros((batch_size, max_outputs, 4))
  f_scores = np.zeros((batch_size, max_outputs))
  f_labels = np.zeros((batch_size, max_outputs), dtype=int)

  for idx, pack in enumerate(map(_inner, np.arange(batch_size))):
    f_counts[idx], f_bboxes[idx], f_scores[idx], f_labels[idx] = pack

  # Return the filtered structures
  return {
      "counts": jnp.array(f_counts, dtype=jnp.int32),
      "bboxes": jnp.array(f_bboxes),
      "scores": jnp.array(f_scores),
      "labels": jnp.array(f_labels, dtype=jnp.int32)
  }
