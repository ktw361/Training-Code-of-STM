import os
import os.path as osp
import numpy as np
import glob
import cv2
from PIL import Image

import torch
import torchvision
from torch.utils import data

direction_m = False

class DAVIS_MO_Test(data.Dataset):
    # for multi object, do shuffling

    def __init__(self, 
                 root, 
                 imset='2017/train.txt', 
                 resolution='480p', 
                 single_object=False,
                 image_resize=(480, 854),
                 calc_num_frames=False,
                 load_from_external=None):
        """_summary_

        Args:
            calc_num_frames (bool, optional):
            load_from_external (str, optional): 
                For EPIC, this points to <epic_rgb_root>
        """
        self.root = root
        self.mask_dir = os.path.join(root, 'Annotations', resolution)
        self.mask480_dir = os.path.join(root, 'Annotations', '480p')
        self.image_dir = os.path.join(root, 'JPEGImages', resolution)
        _imset_dir = os.path.join(root, 'ImageSets')
        _imset_f = os.path.join(_imset_dir, imset)

        self.image_resize = image_resize
        self.load_from_external = load_from_external

        self.videos = []
        self.num_frames = {}
        self.num_objects = {}
        self.shape = {}
        self.size_480p = {}
        with open(os.path.join(_imset_f), "r") as lines:
            for line in lines:
                _video = line.rstrip('\n')
                self.videos.append(_video)
                if calc_num_frames:
                    frame_inds = [
                        int(v.split('.')[0].split('_')[-1]) for v in 
                        os.listdir(osp.join(self.mask_dir, str(_video)))
                    ]
                    start_frame, end_frame = min(frame_inds), max(frame_inds)
                    self.num_frames[_video] = end_frame - start_frame + 1
                else:
                    self.num_frames[_video] = len(glob.glob(os.path.join(self.image_dir, _video, '*.jpg')))
                #print(self.num_frames[_video])
                _mask = np.array(Image.open(sorted(glob.glob(os.path.join(self.mask_dir, _video, '*.png')),reverse=direction_m)[0]).convert("P"))
                #print(sorted(glob.glob(os.path.join(self.mask_dir, _video, '*.png')))[0])
                self.num_objects[_video] = np.max(_mask)
                #print(self.num_objects[_video])
                self.shape[_video] = np.shape(_mask)
                _mask_480 = np.array(Image.open(sorted(glob.glob(os.path.join(self.mask480_dir, _video, '*.png')),reverse=direction_m)[0]).convert("P"))
                self.size_480p[_video] = np.shape(_mask_480)

        self.K = 15
        self.single_object = single_object

    def __len__(self):
        return len(self.videos)


    def To_onehot(self, mask):
        M = np.zeros((self.K, mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for k in range(self.K):
            M[k] = (mask == k).astype(np.uint8)
        return M
    
    def All_to_onehot(self, masks):
        Ms = np.zeros((self.K, masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
        for n in range(masks.shape[0]):
            Ms[:,n] = self.To_onehot(masks[n])
        return Ms

    def __getitem__(self, index):
        video = self.videos[index]
        info = {}
        info['name'] = video
        info['num_frames'] = self.num_frames[video]
        info['size_480p'] = self.size_480p[video]
        info['start_frame'] = int(sorted(glob.glob(os.path.join(self.mask_dir, video, '*.png')),reverse= direction_m)[0].split("/")[-1][-14:-4])

        N_frames = np.empty((self.num_frames[video],)+self.shape[video]+(3,), dtype=np.float32)
        N_masks = np.empty((self.num_frames[video],)+self.shape[video], dtype=np.uint8)
        num_objects = torch.LongTensor([int(self.num_objects[video])])
        return num_objects, info

    def load_single_image(self, video, f, start_frame):
        """ 
        Args:
            f: int, absolute frame index
        """
        N_frames = np.empty((1,)+self.shape[video]+(3,), dtype=np.float32)
        img_file = self.load_single_image_name(
            video, f, start_frame=start_frame, return_base=False)
        img = np.asarray(Image.open(img_file).convert('RGB'))
        img_h, img_w = self.image_resize
        img = cv2.resize(img, (img_w, img_h))
        N_frames[0] = img/255.
        Fs = torch.from_numpy(np.transpose(N_frames.copy(), (3, 0, 1, 2)).copy()).float()
        return Fs

    def load_single_mask(self, video, i):
        N_masks = np.empty((1,)+self.shape[video], dtype=np.uint8)
        mask_file = self.load_single_mask_name(video, i)
        N_masks[0] = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
        Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
        return Ms      

    def load_single_image_name(self, video, f, start_frame, return_base=False):
        """
        Args:
            f: absolute frame index
        """
        if self.load_from_external:
            pid, sub, _ = video.split('_')
            vid = '_'.join([pid, sub])
            file_name_jpg = osp.join(self.load_from_external, pid, vid, f"frame_{f:010d}.jpg")
        else:
            file_name_jpg = sorted(glob.glob(os.path.join(self.image_dir, video,"*.jpg")),reverse=direction_m)[f-start_frame]

        if return_base:
            return file_name_jpg.split("/")[-1][:-4]
        else:
            return file_name_jpg

    def load_single_mask_name(self, video, i):
        """

        Args:
            i: relative frame index
        """
        file_name_jpg = sorted(glob.glob(os.path.join(self.mask480_dir, video,"*.png")),reverse=direction_m)[i]
        return file_name_jpg

if __name__ == '__main__':
    pass
