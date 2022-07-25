"""
Contains functions with different data transforms
"""

from typing import Sequence, Tuple

import numpy as np
import torchvision.transforms as transforms


def get_fundamental_transforms(inp_size: Tuple[int, int]) -> transforms.Compose:
    """Returns the core transforms necessary to feed the images to our model.
    Args:
        inp_size: tuple denoting the dimensions for input to the model

    Returns:
        fundamental_transforms: transforms.compose with the fundamental transforms
    """
    fundamental_transforms = None
    ###########################################################################
    # Student code begins
    ###########################################################################

    fundamental_transforms = transforms.Compose([
        transforms.Resize((inp_size[0], inp_size[1])), 
        transforms.ToTensor()
        ])

    # raise NotImplementedError(
    #     "`get_fundamental_transforms` function in "
    #     + "`data_transforms.py` needs to be implemented"
    # )

    ###########################################################################
    # Student code ends
    ###########################################################################
    return fundamental_transforms


def get_fundamental_augmentation_transforms(
    inp_size: Tuple[int, int]
) -> transforms.Compose:
    """Returns the data augmentation + core transforms needed to be applied on the train set.
    Suggestions: Jittering, Flipping, Cropping, Rotating.
    Args:
        inp_size: tuple denoting the dimensions for input to the model

    Returns:
        aug_transforms: transforms.compose with all the transforms
    """
    fund_aug_transforms = None
    ###########################################################################
    # Student code begin
    ###########################################################################

    fund_aug_transforms = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), 
        # transforms.RandomCrop(inp_size[0], pad_if_needed=True, fill=0, padding_mode='constant'), 
        transforms.RandomHorizontalFlip(p=0.5), 
        # transforms.RandomVerticalFlip(p=0.5),
        # transforms.RandomRotation((-360, 360), resample=False, expand=False, center=None),
        transforms.Resize((inp_size[0], inp_size[1])), 
        transforms.ToTensor(),
        ])

    # raise NotImplementedError(
    #     "`get_fundamental_augmentation_transforms` function in "
    #     + "`data_transforms.py` needs to be implemented"
    # )

    ###########################################################################
    # Student code end
    ###########################################################################
    return fund_aug_transforms


def get_fundamental_normalization_transforms(
    inp_size: Tuple[int, int], pixel_mean: Sequence[float], pixel_std: Sequence[float]
) -> transforms.Compose:
    """Returns the core transforms necessary to feed the images to our model alomg with
    normalization.

    Args:
        inp_size: tuple denoting the dimensions for input to the model
        pixel_mean: image channel means, over all images of the raw dataset
        pixel_std: image channel standard deviations, for all images of the raw dataset

    Returns:
        fundamental_transforms: transforms.compose with the fundamental transforms
    """
    fund_norm_transforms = None
    ###########################################################################
    # Student code begins
    ###########################################################################

    fund_norm_transforms = transforms.Compose([
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.Resize((inp_size[0], inp_size[1])),
        transforms.ToTensor(),
        transforms.Normalize(mean=pixel_mean, std=pixel_std)
    ])

    # raise NotImplementedError(
    #     "`get_fundamental_normalization_transforms` function in "
    #     + "`data_transforms.py` needs to be implemented"
    # )

    ###########################################################################
    # Student code ends
    ###########################################################################
    return fund_norm_transforms


def get_all_transforms(
    inp_size: Tuple[int, int], pixel_mean: Sequence[float], pixel_std: Sequence[float]
) -> transforms.Compose:
    """Returns the data augmentation + core transforms needed to be applied on the train set,
    along with normalization. This should just be your previous method + normalization.
    Suggestions: Jittering, Flipping, Cropping, Rotating.
    Args:
        inp_size: tuple denoting the dimensions for input to the model
        pixel_mean: image channel means, over all images of the raw dataset
        pixel_std: image channel standard deviations, for all images of the raw dataset

    Returns:
        aug_transforms: transforms.compose with all the transforms
    """
    all_transforms = None
    ###########################################################################
    # Student code begins
    ###########################################################################

    all_transforms = transforms.Compose([ 
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), 
        # transforms.RandomCrop(inp_size[0], pad_if_needed=True, fill=0, padding_mode='constant'), 
        transforms.RandomHorizontalFlip(p=0.5), 
        # transforms.RandomVerticalFlip(p=0.5),
        # transforms.RandomRotation((-360, 360), resample=False, expand=False, center=None),
        transforms.Resize(size = (inp_size[0], inp_size[1])),
        transforms.ToTensor(),
        transforms.Normalize(mean=pixel_mean, std=pixel_std)
        ])


    # raise NotImplementedError(
    #     "`get_all_transforms` function in "
    #     + "`data_transforms.py` needs to be implemented"
    # )

    ###########################################################################
    # Student code ends
    ###########################################################################
    return all_transforms
