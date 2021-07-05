#------------------------------------------------------------------------------------
# DT-MIL
# Copyright (c) 2021 Tencent. All Rights Reserved.
# ------------------------------------------------------------------------------------

"""
extract_feature.py will extract offline features from patches
- WSI id1
    - patch1 feat.pkl
    - patch2 feat.pkl
- WSI id2
    - patch1 feat.pkl
    - patch2 feat.pkl
This script will merge the offline patch features within same WSI into one file
- WSI id1.pkl
- WSI id2.pkl
"""

import os
import os.path as osp

import pickle
from glob import glob
from multiprocessing import Pool

from rich import print
from rich.progress import track

select_patch_name = set()

select_patch_dir = './data/test_patches'

file_name = glob(osp.join(select_patch_dir, '*.txt'))
for fp in file_name:
    with open(fp, 'rt') as infile:
        d = infile.readlines()
        d = [x.strip() for x in d]
        for x in d:
            select_patch_name.add(osp.basename(x).rsplit('.', 1)[0].lower())

feat_save_dirx20 = './data/extracted_20_features'
merge_feat_save_dirx20 = './data/extracted_20_feat_merge'

def merge_wsi_feat(wsi_feat_dir) -> None:
    """
    Args:
        wsi_feat_dir: the directory that stores the patch feature of the WSI

    Returns:

    """

    # Search for all the extracted patch features for each WSI id
    files = glob(osp.join(wsi_feat_dir, '*.pkl'))

    files_filter = [x for x in files if osp.basename(x).rsplit('.', 1)[0].lower() in select_patch_name]
    if len(files) != len(files_filter):
        print(f'Filtered {len(files)} => {len(files_filter)}')

    files = files_filter

    save_obj = []
    for fp in files:

        try:
            with open(fp, 'rb') as infile:
                obj = pickle.load(infile)

            # Save Patch name
            obj['feat_name'] = osp.basename(fp).rsplit('.', 1)[0]
            save_obj.append(obj)
        except Exception as e:
            print(f'Error in {fp} as {e}')
            continue

    bname = osp.basename(wsi_feat_dir).lower()  # wsi id
    save_fp = osp.join(merge_feat_save_dir, f'{bname}.pkl')
    with open(save_fp, 'wb') as outfile:
        pickle.dump(save_obj, outfile)

if __name__ == '__main__':
    for feat_save_dir, merge_feat_save_dir in zip(
        [feat_save_dirx20, ],
        [merge_feat_save_dirx20]
    ):
        print(f'Save to {merge_feat_save_dir}')
        os.makedirs(merge_feat_save_dir, exist_ok=True)
        wsi_dirs = glob(osp.join(feat_save_dir, '*'))

        with Pool(80) as p:
            for _ in track(p.imap_unordered(merge_wsi_feat, wsi_dirs), total=len(wsi_dirs)):
                pass
