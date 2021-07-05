
# ------------------------------------------------------------------------
# DT-MIL
# Copyright (c) 2021 Tencent. All Rights Reserved.
# ------------------------------------------------------------------------

import numpy as np
import torch


def check_size(size):
    if type(size) == int:
        size = (size, size)
    if type(size) != tuple:
        raise TypeError('size is int or tuple')
    return size


def center_crop(image, crop_size):
    crop_size = check_size(crop_size)
    h, w, _ = image.shape
    top = (h - crop_size[0]) // 2
    left = (w - crop_size[1]) // 2
    bottom = top + crop_size[0]
    right = left + crop_size[1]
    image = image[top:bottom, left:right, :]
    return image


def random_crop(image, mask, crop_size):
    crop_size = check_size(crop_size)
    h, w, _ = image.shape
    top = np.random.randint(0, h - crop_size[0])
    left = np.random.randint(0, w - crop_size[1])
    bottom = top + crop_size[0]
    right = left + crop_size[1]
    image = image[top:bottom, left:right, :]
    mask = mask[top:bottom, left:right]
    return image, mask


def horizontal_flip(image, mask, rate=0.5):
    if np.random.rand() < rate:
        image = image[:, ::-1, :]
        mask = mask[:, ::-1]
    return image, mask


def vertical_flip(image, mask, rate=0.5):
    if np.random.rand() < rate:
        image = image[::-1, :, :]
        mask = mask[::-1, :]
    return image, mask


def scale_augmentation(image: torch.Tensor, mask, scale_range, crop_size):
    scale_size = np.random.randint(*scale_range)
    image = torch.from_numpy(image.copy())
    mask = torch.from_numpy(mask.copy())
    image = image.permute(2, 0, 1)
    image = image.unsqueeze(0)
    image = torch.nn.functional.interpolate(image, size=scale_size)
    image = image.squeeze(0)
    image = image.permute(1, 2, 0).numpy()

    mask = mask.unsqueeze(0).unsqueeze(0)
    mask = torch.nn.functional.interpolate(mask, size=scale_size).numpy()
    mask = mask.squeeze(0).squeeze(0)

    if scale_size <= crop_size[0]:
        image, mask = pad_image_and_mask(image, mask, target_size=crop_size[0], image_fill_value=0, mask_fill_value=1)
    else:
        image, mask = random_crop(image, mask, crop_size)

    return image, mask


def cutout(image_origin: torch.Tensor, mask_size, mask_value='mean'):

    image = np.copy(image_origin)
    if mask_value == 'mean':
        mask_value = image.mean()
    elif mask_value == 'random':
        mask_value = np.random.randint(0, 256)

    h, w, _ = image.shape
    top = np.random.randint(0 - mask_size // 2, h - mask_size)
    left = np.random.randint(0 - mask_size // 2, w - mask_size)
    bottom = top + mask_size
    right = left + mask_size
    if top < 0:
        top = 0
    if left < 0:
        left = 0
    image[top:bottom, left:right, :].fill(mask_value)
    image = torch.from_numpy(image)
    return image


def pad_image_and_mask(image, mask, target_size, image_fill_value=0, mask_fill_value=1):
    h, w, c = image.shape
    h_pad_upper = (target_size - h) // 2
    h_pad_down = (target_size - h - h_pad_upper)

    w_pad_left = (target_size - w) // 2
    w_pad_right = (target_size - w - w_pad_left)

    ret_img = np.pad(image, ((h_pad_upper, h_pad_down), (w_pad_left, w_pad_right), (0, 0)), 'constant',
                     constant_values=image_fill_value)
    ret_mask = np.pad(mask, ((h_pad_upper, h_pad_down), (w_pad_left, w_pad_right)), 'constant',
                      constant_values=mask_fill_value)

    return ret_img, ret_mask
