#------------------------------------------------------------------------------------
# DT-MIL
# Copyright (c) 2021 Tencent. All Rights Reserved.
# ------------------------------------------------------------------------------------

"""
Extract feature offline
"""
import os, sys, inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import os.path as osp

import random
import cv2
import numpy as np
from glob import glob
import pickle

import torch
from torch import nn
import torch.utils.data as data_utils

from albumentations.pytorch import ToTensorV2
import albumentations as alb
from efficientnet_pytorch import EfficientNet
from rich import progress
from concurrent.futures import ThreadPoolExecutor

tr_trans = alb.Compose([
    alb.Resize(512, 512),
    alb.RandomRotate90(),
    alb.RandomBrightnessContrast(),
    alb.HueSaturationValue(),
    alb.HorizontalFlip(),
    alb.VerticalFlip(),
    alb.CoarseDropout(max_holes=4),
    alb.Normalize(),
    ToTensorV2(),
])

val_trans = alb.Compose([
    alb.Resize(512, 512),
    alb.Normalize(),
    ToTensorV2(),
])


class PatchDataset(data_utils.Dataset):
    def __init__(self, img_fp_list, trans):
        self.img_fp_list = img_fp_list
        self.trans = trans

    def __len__(self):
        return len(self.img_fp_list)

    def __getitem__(self, idx):
        img_fp = self.img_fp_list[idx]
        img = cv2.imread(img_fp)
        if img is None:
            img = np.zeros((224, 224, 3)).astype('uint8')
        img = img[:, :, ::-1]
        pid = osp.basename(osp.dirname(img_fp))
        img_bname = osp.basename(img_fp).rsplit('.', 1)[0]
        aug_img = self.trans(image=img)['image']
        return pid, img_bname, aug_img


def save_feat_in_thread(batch_pid, batch_img_bname, batch_tr_feat, save_dir):
    for b_idx, (pid, img_bname, tr_feat) in enumerate(zip(batch_pid, batch_img_bname, batch_tr_feat)):
        feat_save_dir = osp.join(save_dir, pid)
        os.makedirs(feat_save_dir, exist_ok=True)
        feat_save_name = osp.join(feat_save_dir, img_bname+'.pkl')

        if osp.exists(feat_save_name):
            try:
                with open(feat_save_name, 'rb') as infile:
                    save_dict = pickle.load(infile)
            except:
                save_dict = {}
        else:
            save_dict = {}

        tr_aug_feat = np.stack([tr_feat])
        if 'tr' in save_dict.keys():
            save_dict['tr'] = np.concatenate([tr_aug_feat, save_dict['tr']])
        else:
            save_dict['tr'] = tr_aug_feat

        with open(feat_save_name, 'wb') as outfile:
            pickle.dump(save_dict, outfile)


def save_val_feat_in_thread(batch_pid, batch_img_bname, batch_val_feat, save_dir):
    for b_idx, (pid, img_bname, val_feat) in enumerate(zip(batch_pid, batch_img_bname, batch_val_feat)):
        feat_save_dir = osp.join(save_dir, pid)
        os.makedirs(feat_save_dir, exist_ok=True)
        feat_save_name = osp.join(feat_save_dir, img_bname+'.pkl')

        if osp.exists(feat_save_name):
            with open(feat_save_name, 'rb') as infile:
                save_dict = pickle.load(infile)
        else:
            save_dict = {}

        save_dict['val'] = val_feat

        with open(feat_save_name, 'wb') as outfile:
            pickle.dump(save_dict, outfile)


def pred_and_save_with_dataloader(model, img_fp_list, save_dir):
    random.seed(42)
    model.eval()
    executor = ThreadPoolExecutor(max_workers=48)
    dl = torch.utils.data.DataLoader(
        PatchDataset(img_fp_list, trans=tr_trans),
        batch_size=1,
        num_workers=1,
        shuffle=False,
        drop_last=False
    )

    val_dl = torch.utils.data.DataLoader(
        PatchDataset(img_fp_list, trans=val_trans),
        batch_size=1,
        num_workers=1,
        shuffle=False,
        drop_last=False
    )

    for batch in progress.track(val_dl):
        batch_pid, batch_img_bname, batch_tr_img = batch
        batch_tr_img = batch_tr_img.cuda()

        with torch.no_grad():
            val_feat = model(batch_tr_img)
            val_feat = val_feat.cpu().numpy()
            executor.submit(save_val_feat_in_thread, batch_pid, batch_img_bname, val_feat, save_dir)

    for batch in progress.track(dl):
        batch_pid, batch_img_bname, batch_tr_img = batch
        batch_tr_img = batch_tr_img.cuda()

        with torch.no_grad():
            tr_feat = model(batch_tr_img)
            tr_feat = tr_feat.cpu().numpy()
            executor.submit(save_feat_in_thread, batch_pid, batch_img_bname, tr_feat, save_dir)


efn_pretrained = {
    'efficientnet-b0': './checkpoints/efficientnet-b0-355c32eb.pth',
}


class EffNet(nn.Module):
    def __init__(self, efname='efficientnet-b0'):
        super(EffNet, self).__init__()
        self.model = EfficientNet.from_name(efname)
        pretrain_model_fp = efn_pretrained[efname]
        print('Load pretrain model from ' + pretrain_model_fp)
        self.model.load_state_dict(torch.load(pretrain_model_fp))

    def forward(self, data):
        bs = data.shape[0]
        feat = self.model.extract_features(data)
        feat = nn.functional.adaptive_avg_pool2d(feat, output_size=(1))
        feat = feat.view(bs, -1)
        return feat


if __name__ == "__main__":
    model = EffNet()
    model = nn.DataParallel(model).cuda()

    img_fp_list = []
    file_dir = './data/test_patches'
    save_dir = './data/extracted_20_features'

    wsi_patch_info_dir = file_dir
    save_dir = save_dir
    bag_fp_list = glob(osp.join(wsi_patch_info_dir, '*.txt'))
    for bag_fp in bag_fp_list:
        with open(bag_fp, 'rt') as infile:
            lines = infile.readlines()
            lines = [x.strip() for x in lines]
        img_fp_list.extend(lines)

    print('Len of img '.format(len(img_fp_list)))

    img_fp_list = sorted(img_fp_list)
    pred_and_save_with_dataloader(model, img_fp_list, save_dir=save_dir)