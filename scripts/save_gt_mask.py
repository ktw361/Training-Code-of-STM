"""
This script saves available masks.
"""

from libzhifan import io
from epic_frame_predict import EpicFramePredictor
import tqdm
import os
import os.path as osp

data = io.read_json('/home/skynet/Zhifan/htmls/clean_gt_mask/gt_frames.json')

predictor = EpicFramePredictor(seg_root='/home/skynet/Zhifan/data/more_segs/',
                               epic_rgb_root='/home/skynet/Zhifan/datasets/epic/rgb_root/')

save_dir = '/home/skynet/Zhifan/data/epic_analysis/interpolation/'

for nid, dd in tqdm.tqdm(data.items()):
    vid = '_'.join(nid.split('_')[:2])
    frames = dd['frames']
    dst_root = osp.join(save_dir, vid)
    if not osp.exists(dst_root):
        os.makedirs(dst_root)
    # print(nid)
    for frame in frames:
        mask_th = predictor._read_mask(vid, frame)
        dst = osp.join(dst_root, f'frame_{frame:010d}.png')
        if osp.exists(dst):
            continue
        predictor._save_annotated_mask(mask_th, dst)
        # print(dst)
