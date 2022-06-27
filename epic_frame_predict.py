import argparse
import tqdm
import os
import os.path as osp
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import logging

from model.model import STM
from epic_seg import EpicSegGT, read_epic_image


class EpicFramePredictor:

    def __init__(self,
                 seg_root,
                 epic_rgb_root):
        self.seg_reader = EpicSegGT(seg_root, hei=1080, wid=1920)
        self.epic_rgb_root = epic_rgb_root
        self.model = self._load_model()
        self.num_objects = len(self.seg_reader.all_cats)  # TODO(zhifan): confirm
        self.num_objects = torch.LongTensor([int(self.num_objects)])
        self.image_resize = (480, 854)
        PALETTE = '/home/skynet/Zhifan/datasets/EPIC_DAVIS/Annotations/480p/P11_105_0000/0023_P11_105_frame_0000000242.png'
        self.palette = Image.open(PALETTE).getpalette()

    def _load_model(self,
                    backbone='resnet50',
                    pth='../davis_weights/epic_dense_epic_0skip_sparse_resnet50_100000_32_29999.pth',
                    ):
        model = nn.DataParallel(STM(backbone))
        if torch.cuda.is_available():
            model.cuda()
        model.eval()

        model.load_state_dict(torch.load(pth))
        return model

    def _read_frame(self, vid, frame):
        """

        Args:
            vid (_type_): _description_
            frame (_type_): _description_

        Returns:
            torch.Tensor with shape (1, 3, 480, 854)
        """
        img = np.asarray(read_epic_image(
            vid, frame, root=self.epic_rgb_root, as_pil=True).convert('RGB'))
        img_h, img_w = self.image_resize
        img = cv2.resize(img, (img_w, img_h))
        Fs = torch.from_numpy(img / 255.).float().permute(2, 0, 1).unsqueeze_(0)
        return Fs

    def _read_mask(self, vid, frame):
        """

        Returns:
            torch.Tensor with shape (1, 15, 480, 854)
        """
        mask = self.seg_reader.read_frame(vid, frame)  # (1, num_objects, 480, 854)
        img_h, img_w = self.image_resize
        mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST).astype('uint8')
        mask = np.transpose(mask, [2, 0, 1])
        K = 15
        mask_full = np.zeros([K, img_h, img_w], dtype=mask.dtype)
        mask_full[:self.num_objects.item(), :, :] = mask
        mask_full = torch.from_numpy(mask_full).float().unsqueeze_(0)
        return mask_full

    def _save_annotated_mask(self, mask_th, fname):
        """_summary_

        Args:
            mask_th (torch.Tensor): (1, 15, 480, 854)
        """
        _, C, H, W = mask_th.shape
        mask = np.zeros([H, W], dtype=np.uint8)
        for c in range(1, C):
            nz = mask_th[0, c, :, :] == 1
            mask[nz] = c
        mask_E = Image.fromarray(mask)
        mask_E.putpalette(self.palette)
        mask_E.save(fname)

    def interpolate_frames(self,
                           vid: str,
                           frame_interp: int,
                           save_dir: str,
                           Mem_every=None,
                           Mem_number=5):
        """
        Interpolate (all) intermediate frames from [st, ed],
            where st and ed are frames with ground truth mask according to annotation.

        Mask will be read as torch.Tensor with shape (1, C, H, W)

        Interpolated files are saved into `save_dir`,
        e.g.
        save_dir/
            P11_105/
                frame_0000000200.png
                ...
                frame_0000000300.png

        """
        out = self.seg_reader.locate_input_pair(vid, frame_interp)
        assert out is not None
        st, ed = out
        num_frames = ed - st - 1  # st and ed are existing mask

        if Mem_every:
            to_memorize = [int(i) for i in np.arange(st+1, ed, step=Mem_every)]
        elif Mem_number:
            to_memorize = [int(round(i)) for i in np.linspace(
                st+1, ed, num=Mem_number+2)[:-1]]
        else:
            raise NotImplementedError

        test_path = osp.join(save_dir, vid)
        if not osp.exists(test_path):
            os.makedirs(test_path)

        F_last = self._read_frame(vid, st)  # (1, 3, 480, 854)
        M_last = self._read_mask(vid, st)  # (1, num_objects, 480, 854)
        E_last = M_last

        F_last_rev = self._read_frame(vid, ed)
        M_last_rev = self._read_mask(vid, ed)
        E_last_rev = M_last_rev

        self._save_annotated_mask(M_last, osp.join(test_path, f'frame_{st:010d}.png'))
        self._save_annotated_mask(M_last_rev, osp.join(test_path, f'frame_{ed:010d}.png'))

        _first_path = osp.join(test_path, f'frame_{st+1:010d}.png')
        _last_path = osp.join(test_path, f'frame_{ed-1:010d}.png')
        if osp.exists(_first_path) and osp.exists(_last_path):
            logging.info(f"{_first_path} and {_last_path} already exist, skipping.")
            return
        #print("NUM FRAMES:",num_frames)

        pbar = tqdm.tqdm(total=num_frames)
        for t in range(st + 1, ed):
            pbar.update(1)
            #print('current_frame: {},num_frames: {}, num_objects: {}'.format(t, num_frames, num_objects.numpy()[0]))

            # memorize
            with torch.no_grad():
                prev_key, prev_value = self.model(F_last, E_last, torch.tensor([self.num_objects]))

            # if t-1 == 0: #
            if t == st + 1:
                this_keys, this_values = prev_key, prev_value # only prev memory
                #then add the frame of the end of the seq.
                with torch.no_grad():
                    prev_key_rev, prev_value_rev = self.model(F_last_rev, E_last_rev, torch.tensor([self.num_objects]))

                this_keys = torch.cat([this_keys, prev_key_rev], dim=3)
                this_values = torch.cat([this_values, prev_value_rev], dim=3)
            else:
                this_keys = torch.cat([keys, prev_key], dim=3)
                this_values = torch.cat([values, prev_value], dim=3)

            F_ = self._read_frame(vid, t)

            # segment
            with torch.no_grad():
                logit = self.model(F_, this_keys, this_values, torch.tensor([self.num_objects]))
            E = F.softmax(logit, dim=1)

            # update
            if t in to_memorize:
                keys, values = this_keys, this_values
            pred = torch.argmax(E[0], dim=0).cpu().numpy().astype(np.uint8)
            F_last = F_

            img_E = Image.fromarray(pred)
            img_E.putpalette(self.palette)
            img_E.save(os.path.join(test_path, f"frame_{t:010d}.png"))  #### there to save the images

        pbar.close()


def predict_related_frames(annotation_file,
                           save_dir,
                           seg_root,
                           epic_rgb_root):
    """
    E.g.
        P01_01_100 cup right 28802

    Args:
        annotation_file (_type_): _description_
    """
    predictor = EpicFramePredictor(
        seg_root=seg_root,
        epic_rgb_root=epic_rgb_root)

    with open(annotation_file) as fp:
        lines = fp.readlines()
        lines = [v.strip().replace('\t', ' ') for v in lines]

    for line in tqdm.tqdm(lines):
        nid, frame = [v for v in line.split(' ') if len(v) > 0]
        vid = '_'.join(nid.split('_')[:2])
        frame = int(frame)
        predictor.interpolate_frames(vid, frame, save_dir)


def get_interpolation_overlay(video_id, frame_idx, save_dir):
    """ Put image and mask side-by-side, save into directory """
    img = read_epic_image(video_id, frame_idx)
    mask = f"{save_dir}/{video_id}/frame_{frame_idx:010d}.png"
    mask_pil = Image.open(mask)
    mask = np.asarray(mask_pil)
    h, w = mask.shape
    img = cv2.resize(img, (w, h))
    canvas = np.zeros([h, 2*w, 3], dtype=np.uint8)
    canvas[:, :w, :] = img.copy()
    palette = mask_pil.getpalette()
    for c in np.unique(mask[mask != 0]):
        img[mask == c, :] = palette[3*c:3*c+3]
    canvas[:, w:, :] = img
    return canvas


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_file', default='~/data/epic_analysis/clean_tiny_gt.txt')
    parser.add_argument('--save_dir', default='./interpolation')

    parser.add_argument('--seg_root', default='/home/skynet/Zhifan/data/more_segs')
    parser.add_argument('--epic_rgb_root', default='/home/skynet/Zhifan/data/epic_rgb_frames/')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    predict_related_frames(args.ann_file,
                           args.save_dir,
                           args.seg_root,
                           args.epic_rgb_root)
