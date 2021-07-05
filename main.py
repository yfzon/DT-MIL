# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------
# DT-MIL
# Copyright (c) 2021 Tencent. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

import argparse
import random
from pathlib import Path
import os
import os.path as osp
import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset
from engine import evaluate

from models import build_wsi_feat
import pprint



def get_args_parser():
    parser = argparse.ArgumentParser('Deformable Transformer MIL', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    parser.add_argument('--backbone', default='resnet18', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")

    parser.add_argument('--num_feature_levels', default=1, type=int, help='number of feature levels')

    parser.add_argument('--num_class', default=2, type=int, help='number of class')
    parser.add_argument('--map_size', default=22, type=int, help='size of merge feature map')
    parser.add_argument('--scale', default='x20', type=str, help='select scale')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")

    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    parser.add_argument('--output_dir', default='./results',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=6, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    # custom param
    parser.add_argument('--num_input_channels', default=1280, type=int)
    parser.add_argument('--out_channels', default=256, type=int)
    parser.add_argument('--feat_map_size', default=60, type=int)  # size of featuremap

    parser.add_argument('--model_name', default='transformer', type=str)  #
    parser.add_argument('--drop_decoder', action='store_true')

    return parser


def main(args):
    utils.init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


    print(f'Build Model: {args.model_name}')
    if args.model_name == 'transformer':
        model, criterion = build_wsi_feat(args)
        model.to(device)
    elif args.model_name == 'pure_transformer':
        from models import build_pure_transformer
        model, criterion = build_pure_transformer(args)
        model.to(device)
    else:
        raise ValueError(f'Model name not found')

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    dataset_test = build_dataset(image_set='test', args=args)

    if args.distributed:
        if args.cache_mode:
            sampler_test = samplers.NodeDistributedSampler(dataset_test, shuffle=False)
        else:
            sampler_test = samplers.DistributedSampler(dataset_test, shuffle=False)
    else:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test,
                                  drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                  pin_memory=True)


    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module


    checkpoint = torch.load(args.frozen_weights, map_location='cpu')

    missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
    unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
    if len(missing_keys) > 0:
        print('Missing Keys: {}'.format(missing_keys))
    if len(unexpected_keys) > 0:
        print('Unexpected Keys: {}'.format(unexpected_keys))

    time_now = datetime.datetime.now()
    args.output_dir = osp.join(args.output_dir, f'{time_now.year}_{time_now.month}_{time_now.day}_{time_now.hour}_{time_now.minute}')
    os.makedirs(args.output_dir, exist_ok=True)

    args_dict = vars(args)
    with open(osp.join(args.output_dir, 'params.txt'), 'wt') as outfile:
        pprint.pprint(args_dict, indent=4, stream=outfile)

    eval_result = evaluate(model, criterion,
                                       data_loader_test, device, args.output_dir,
                                       args.distributed,
                                       display_header="Test",
                                       is_last_eval=True,
                                       save_path=args.output_dir
                                       )


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DT-MIL inference script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
