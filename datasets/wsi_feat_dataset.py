# ------------------------------------------------------------------------
# DT-MIL
# Copyright (c) 2021 Tencent. All Rights Reserved.
# ------------------------------------------------------------------------

import torch
import os.path as osp
import torch.utils.data as data_utils
from util.misc import get_local_rank, get_local_size

import re
from typing import Tuple
import numpy as np
import pickle
from util import numpy_img_aug as tensor_aug


def parse_patch_fname(fp: str) -> Tuple[int, int]:
    # d15-08498-tile-r16896-c20480-512x512
    fp = osp.basename(fp).rsplit('.', 1)[0]
    psize = int(fp.rsplit('x', 1)[-1])
    r_loc = int(re.findall(r'(?<=r)\d+', fp)[0])
    c_loc = int(re.findall(r'(?<=c)\d+', fp)[0])

    r_loc = r_loc // psize
    c_loc = c_loc // psize
    return (r_loc, c_loc)


class WSIFeatDataset(data_utils.Dataset):
    """
    Load Feat
    """

    def __init__(self, ann_file=None, transforms=None, num_class=2, is_training=False, feat_map_size=60,
                 cache_mode=False, local_rank=0, local_size=1):
        super().__init__()

        with open(ann_file, 'rt') as infile:
            data = infile.readlines()

        self.wsi_loc = []
        for each in data:

            pid, feat_fp, label = each.strip().split(',')
            label = int(label)
            self.wsi_loc.append((pid, feat_fp, label))

        # sort by WSI id
        sorted(self.wsi_loc, key=lambda x: x[0])

        self.num_class = num_class

        self.pid_2_img_id = {}
        for idx, v in enumerate(self.wsi_loc):
            self.pid_2_img_id[v[0]] = idx

        self._transforms = transforms
        self.feat_map_size = feat_map_size
        self.is_training = is_training

    def __len__(self):
        return len(self.wsi_loc)

    def __getitem__(self, index):
        image_id, patch_feat_fp, label = self.wsi_loc[index]

        label = torch.tensor(label > 0).long()

        patch_loc = []

        with open(patch_feat_fp, 'rb') as infile:
            patch_feat = pickle.load(infile)

        for each_ob in patch_feat:
            feat_name = each_ob['feat_name']
            patch_loc.append(parse_patch_fname(feat_name))

        try:
            num_channels = patch_feat[0]['val'].shape[-1]
            min_r = min([x[0] for x in patch_loc])
            min_c = min([x[1] for x in patch_loc])

            patch_loc = [(x[0] - min_r, x[1] - min_c) for x in patch_loc]

            max_r = max([x[0] for x in patch_loc])
            max_c = max([x[1] for x in patch_loc])
        except:
            raise ValueError(f'Error in {image_id} at {patch_feat_fp}')

        feat_map = np.zeros((max_r + 1, max_c + 1, num_channels))

        for (r_idx, c_idx), each_ob in zip(patch_loc, patch_feat):
            if self.is_training and 'tr' in each_ob.keys():
                c_feat = each_ob['tr']
                select_idx = np.random.choice(c_feat.shape[0], 1).item()
                select_feat = c_feat[select_idx]
                feat_map[r_idx, c_idx, :] = select_feat.reshape(-1)
            else:
                select_feat = each_ob['val']
                feat_map[r_idx, c_idx, :] = select_feat.reshape(-1)

        # samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        h, w = feat_map.shape[:2]

        if h > self.feat_map_size:
            feat_map = tensor_aug.center_crop(feat_map, (self.feat_map_size, w))
        if w > self.feat_map_size:
            feat_map = tensor_aug.center_crop(feat_map, (h, self.feat_map_size))

        mask = np.zeros(feat_map.shape[:2])
        feat_map, mask = tensor_aug.pad_image_and_mask(feat_map, mask, target_size=self.feat_map_size,
                                                       image_fill_value=0, mask_fill_value=1)

        feat_map, mask = self._transforms(feat_map, mask)

        feat_map = torch.from_numpy(feat_map.copy()).float()
        mask = torch.from_numpy(mask.copy())

        feat_map = feat_map.permute(2, 0, 1)
        mask = mask.bool()

        pid_img_id = self.pid_2_img_id[image_id]
        target = {
            'label': label,
            'pid': torch.tensor(pid_img_id)
        }

        return feat_map, mask, target


class TrFeatureMapAug:
    def __call__(self, feat, mask, p=0.5):
        feat, mask = tensor_aug.horizontal_flip(feat, mask)
        feat, mask = tensor_aug.vertical_flip(feat, mask)

        if np.random.rand() < p:
            scale_range_max = feat.shape[0] * 1.2
            scale_range_min = feat.shape[0] * 0.8
            crop_size = feat.shape[:2]
            scale_range = (scale_range_min, scale_range_max)
            feat, mask = tensor_aug.scale_augmentation(feat, mask, scale_range=scale_range, crop_size=crop_size)

        if np.random.rand() < p:
            mask_size = feat.shape[0] // 8
            feat = tensor_aug.cutout(feat, mask_size=mask_size)

        rot_k = np.random.randint(0, 4)

        feat = np.rot90(feat, k=rot_k, axes=(0, 1))
        mask = np.rot90(mask, k=rot_k, axes=(0, 1))

        return feat, mask


class ValFeatureMapAug:
    def __call__(self, feat, mask):
        return feat, mask


def make_wsi_transforms(image_set):
    if image_set == 'train':
        return TrFeatureMapAug()
    else:
        return ValFeatureMapAug()


def build(image_set, args):

    if args.scale == 'x10':
        ann_file = f'./data/{image_set}.txt'
    elif args.scale == 'x20':
        ann_file = f'./data/x20{image_set}.txt'

    print(f'Build dataset : {args.scale} with num class {args.num_class}')
    dataset = WSIFeatDataset(transforms=make_wsi_transforms(image_set),
                             ann_file=ann_file,
                             num_class=args.num_class,
                             cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size(),
                             feat_map_size=args.feat_map_size)

    return dataset
