# ------------------------------------------------------------------------
# DT-MIL
# Copyright (c) 2021 Tencent. All Rights Reserved.
# ------------------------------------------------------------------------


def build_dataset(image_set, args):

    from .wsi_feat_dataset import build as build_wsi_feat_dataset
    return build_wsi_feat_dataset(image_set, args)

