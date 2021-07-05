# ------------------------------------------------------------------------
# DT-MIL
# Copyright (c) 2021 Tencent. All Rights Reserved.
# ------------------------------------------------------------------------


import torch
from torch import Tensor
from util.misc import NestedTensor


def convert_feature_seq2_map(feature_seq: Tensor, loc: Tensor, seq_mask: Tensor, map_size: int, is_train=False):
    bs = loc.size(0)
    seq_len = loc.size(1)
    out = []
    for scale_name, scale_feature in sorted(feature_seq.items()):

        scale_feature = torch.mean(scale_feature, dim=(2, 3), keepdim=True)

        _, channel, h, w = scale_feature.shape
        scale_feature = scale_feature.view(-1, seq_len, channel, h, w)

        ret_map = torch.zeros((bs, channel, h * map_size, w * map_size)).to(scale_feature.device)
        padding_mask = torch.zeros((bs, h * map_size, w * map_size)).to(scale_feature.device)

        b_list = []
        s_list = []
        r_list = []
        c_list = []
        filled = False
        for b_idx in range(bs):
            for seq_idx in range(seq_len):

                r, c = loc[b_idx, seq_idx]
                r, c = int(r), int(c)
                if c >= map_size or r >= map_size:
                    continue

                b_list.append(b_idx)
                s_list.append(seq_idx)
                r_list.append(r)
                c_list.append(c)
                filled = True
                ret_map[b_idx, :, r * h:(r + 1) * h, c * w:(c + 1) * w] = ret_map[b_idx, :, r * h:(r + 1) * h,
                                                                          c * w:(c + 1) * w] + scale_feature[
                                                                              b_idx, seq_idx]

                padding_mask[b_idx, r * h:(r + 1) * h, c * w:(c + 1) * w] = 1

        if not filled:
            print(loc[:30])

        padding_mask = padding_mask.float()
        ret_map.to(scale_feature.device)
        padding_mask.to(scale_feature.device)

        padding_mask = padding_mask.bool()

        nt = NestedTensor(ret_map, padding_mask).to(scale_feature.device)
        out.append(nt)

    return out
