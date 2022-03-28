"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os


def display_mask(mask):
    dict_col = np.array(
        [
            [0, 0, 0],
            [0, 255, 0],
            [255, 0, 0],
            [0, 0, 255],
            [255, 255, 255],
            [96, 96, 96],
            [253, 96, 96],
            [255, 255, 0],
            [237, 127, 16],
            [102, 0, 153],
        ]
    )

    dict_col = np.array(
        [
            [0, 0, 0],  # black
            [0, 255, 0],  # green
            [255, 0, 0],  # red
            [0, 0, 255],  # blue
            [0, 255, 255],  # cyan
            [255, 255, 255],  # white
            [96, 96, 96],  # grey
            [255, 255, 0],  # yellow
            [237, 127, 16],  # orange
            [102, 0, 153],  # purple
            [88, 41, 0],  # brown
            [253, 108, 158],  # pink
            [128, 0, 0],  # maroon
            [255, 0, 255],
            [255, 0, 127],
            [0, 128, 255],
            [0, 102, 51],  # 17
            [192, 192, 192],
            [128, 128, 0],
            [84, 151, 120],
        ]
    )

    try:
        len(mask.shape) == 2
    except AssertionError:
        print("Mask's shape is not 2")
    mask_dis = np.zeros((mask.shape[0], mask.shape[1], 3))
    # print('mask_dis shape',mask_dis.shape)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[0]):
            mask_dis[i, j, :] = dict_col[mask[i, j]]
    return mask_dis


def tensor2im(input_image, imtype=np.uint8):
    """ "Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = (
            image_tensor[0].cpu().float().numpy()
        )  # convert it into a numpy array nb : the first image of the batch is displayed
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        if len(image_numpy.shape) != 2:  # it is an image
            image_numpy.clip(-1, 1)
            image_numpy = (
                (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
            )  # post-processing: transpose and scaling
        else:  # it is  a mask
            image_numpy = image_numpy.astype(np.uint8)
            image_numpy = display_mask(image_numpy)
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name="network"):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print("shape,", x.shape)
    if val:
        x = x.flatten()
        print(
            "mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f"
            % (np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x))
        )


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def gaussian(in_tensor, stddev):
    noisy_image = (
        torch.normal(0, stddev, size=in_tensor.size()).to(in_tensor.device) + in_tensor
    )
    return noisy_image


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


MAX_INT = 1000000000
