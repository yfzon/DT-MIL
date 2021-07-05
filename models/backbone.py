# ------------------------------------------------------------------------
# DT-MIL
# Copyright (c) 2021 Tencent. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Backbone modules.
"""

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, Nested3Tensor

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


import os
import sys

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve


def load_url(url, model_dir='../pretrained', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)

    print(f'load from : {cached_file}')
    return torch.load(cached_file, map_location=map_location)


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        norm_layer = FrozenBatchNorm2d

        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=False, norm_layer=norm_layer)

        state = load_url('resnet18-5c106cde.pth')
        backbone.load_state_dict(state)
        super().__init__(backbone, train_backbone, return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


from util.feature_seq2_feature_map import convert_feature_seq2_map


class WSIJoiner(nn.Sequential):
    def __init__(self, backbone, position_embedding, map_size=22):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels
        self.map_size = map_size

    def forward(self, samples: Nested3Tensor):
        # tensor_list (B, N, C, H, W)
        tensor_list, loc, seq_mask = samples.decompose()
        B, N, C, H, W = tensor_list.shape
        tensor_list = tensor_list.view(-1, C, H, W)
        xs = self[0](tensor_list)

        out = convert_feature_seq2_map(xs, loc=loc, seq_mask=seq_mask, map_size=self.map_size, is_train=self.training)
        pos = []

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


class FeatureMapDrBackbone(nn.Sequential):
    """
    Backbone to reduce feature map dimension
    """

    def __init__(self, num_input_channels, out_channels):
        super(FeatureMapDrBackbone, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Sequential(
            nn.BatchNorm2d(num_input_channels),
            nn.Conv2d(num_input_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )

    def forward(self, tensor_list: NestedTensor):
        feat_map = tensor_list.tensors
        mask = tensor_list.mask

        feat_map = self.conv(feat_map)
        return NestedTensor(feat_map, mask)

    @property
    def num_channels(self):
        return [self.out_channels]


class FeatureMapJoiner(nn.Sequential):
    """
    创建Joiner, 只使用Feature map 作为输入
    """

    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    @property
    def num_channels(self):
        return self[0].num_channels

    def forward(self, tensor_list: NestedTensor):
        # print(f'Wththe')
        xs = {
            'input_feature_map': self[0](tensor_list)
        }
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            # print(x.tensors.shape, x.mask.shape)
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks or (args.num_feature_levels > 1)
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    return model


def build_WSIBackbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks or (args.num_feature_levels > 1)
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = WSIJoiner(backbone, position_embedding, args.map_size)
    return model


def build_WSIFeatureMapBackbone(args):
    position_embedding = build_position_encoding(args)
    backbone = FeatureMapDrBackbone(num_input_channels=args.num_input_channels, out_channels=args.out_channels)
    model = FeatureMapJoiner(backbone, position_embedding)
    return model
