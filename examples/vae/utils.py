import math
import numpy
import jax
import jax.numpy as np


def make_grid(ndarray, nrow=8, padding=2,
              normalize=False, range=None, scale_each=False, pad_value=0.0):
    """Make a grid of images.
    Args:
        ndarray (array_like or list): 4D mini-batch ndarray of shape (B x H x W x C)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by :attr:`range`. Default: ``False``.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.
    """
    if not (isinstance(ndarray, np.DeviceArray) or
            (isinstance(ndarray, list) and all(isinstance(t, np.DeviceArray) for t in ndarray))):
        raise TypeError('array_like or list of tensors expected, got {}'.format(type(ndarray)))

    # if list of tensors, convert to a 4D mini-batch ndarray
    if isinstance(ndarray, list):
        ndarray = np.stack(ndarray, dim=0)

    if ndarray.ndim == 2:  # single image H x W
        ndarray = np.expamd_dims(ndarray, -1)
    if ndarray.ndim == 3:  # single image
        if ndarray.shape[-1] == 1:  # if single-channel, convert to 3-channel
            ndarray = np.concatenate((ndarray, ndarray, ndarray), -1)
        ndarray = np.expamd_dims(ndarray, 0)

    if ndarray.ndim == 4 and ndarray.shape[-1] == 1:  # single-channel images
        ndarray = np.concatenate((ndarray, ndarray, ndarray), -1)

    if ndarray.shape[0] == 1:
        return ndarray.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = ndarray.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(ndarray.shape[1] + padding), int(ndarray.shape[2] + padding)
    num_channels = ndarray.shape[3]
    grid = np.full((height * ymaps + padding, width * xmaps + padding, num_channels), pad_value).astype(np.float32)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid = jax.ops.index_update(grid, jax.ops.index[y * height + padding: (y+1) * height, x * width + padding: (x+1) * width], ndarray[k])
            k = k + 1
    return grid


def save_image(ndarray, fp, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0, format=None):
    """Save a given Tensor into an image file.
    Args:
        ndarray (array_like or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp - A filename(string) or file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    grid = make_grid(ndarray, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)

    ndarr = np.clip(grid * 255.0 + 0.5, 0, 255).astype(np.uint8)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    im = Image.fromarray(ndarr.copy())
    im.save(fp, format=format)
