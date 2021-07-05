# -*- coding: utf-8 -*-

# ------------------------------------------------------------------------
# DT-MIL
# Copyright (c) 2021 Tencent. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------


"""
Evaluation functions used in main.py
"""

from util.gpu_gather import GpuGather
import os

import torch
import util.misc as utils
import pandas as pd
import numpy as np


@torch.no_grad()
def evaluate(model, criterion, data_loader, device, output_dir, is_distributed, display_header="Valid",
             is_last_eval=False, save_path=''):
    model.eval()

    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")

    header = f'{display_header}:'

    gpu_gather = GpuGather(is_distributed=is_distributed)
    print_freq = len(data_loader) // 4
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        label = torch.tensor([x['label'] for x in targets])
        pid = torch.tensor([x['pid'] for x in targets])

        try:
            outputs = model(samples)
        except Exception as e:
            print(samples['target'])
            raise e

        loss = criterion(outputs, label.cuda())

        loss_dict = {'loss': loss}

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)

        metric_logger.update(loss=loss_dict_reduced['loss'].item())

        b = outputs.size(0)
        gpu_gather.update(pred=torch.softmax(outputs.detach().cpu().view(b, -1), dim=1).numpy())
        gpu_gather.update(label=label.cpu().numpy().reshape(-1))
        gpu_gather.update(pid=pid.cpu().numpy().tolist())

    # gather the stats from all processes

    gpu_gather.synchronize_between_processes()

    pred = gpu_gather.pred
    pred = np.concatenate(pred)
    label = gpu_gather.label
    label = np.concatenate(label)
    pid = gpu_gather.pid
    pid = np.array(pid)
    print(f'number of example: {pid.shape}')

    data = np.hstack([label.reshape(-1, 1), pred.reshape(-1, 2)])
    df = pd.DataFrame(data, columns=['target', 'pred_0', 'pred_1'])
    save_fp = os.path.join(save_path, 'pred.csv')
    print(f'Save result to {save_fp}')
    df.to_csv(save_fp, index=False, encoding='utf_8_sig')

    return df
