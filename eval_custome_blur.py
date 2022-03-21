from __future__ import division
import torch
from torch.autograd import Variable
from torch.utils import data

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models

# general libs
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import time
import tqdm
import os
import argparse
import copy


### My libs
from dataset.dataset import DAVIS_MO_Test
from model.model import STM

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from evaldavis2017.davis2017.davis import DAVIS
from evaldavis2017.davis2017.metrics import db_eval_boundary, db_eval_iou
from evaldavis2017.davis2017 import utils
from evaldavis2017.davis2017.results import Results
from scipy.optimize import linear_sum_assignment

def select_lowest_blur (folder, frame_index, blur_window):
    import glob  
    max_blur = 0 #small value
    max_index = frame_index - blur_window # since it + or - blur_window
    all_images = sorted(glob.glob(folder + '/*.jpg'))
    #start_index = max(-blur_window,0)
    #end_index = min(blur_window+1,len(sorted(glob.glob(folder + '/*.jpg'))))
    for i in range(-blur_window,blur_window+1):
        if frame_index + i < 1:
          image_path = all_images[1]
        elif frame_index + i >= (len(all_images) - 2):
          image_path = all_images[(len(all_images) - 2)]
        else:
          image_path = all_images[frame_index + i]
        blur_value = measure_blur_image(image_path)
        #print (blur_value)
        if blur_value > max_blur:
            max_blur = blur_value
            max_index = frame_index + i
    return max_index, max_blur

def measure_blur_image(path): #function to calculate the blur in each one of the sub-folders of a directory, it also displays the blured images
    import cv2
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()
    return blur_value


def Run_video(dataset,video, num_frames,start_frame, num_objects,model,Mem_every=None, Mem_number=None):
    
    import glob
    import os

    # initialize storage tensors
    if Mem_every:
        to_memorize = [int(i) for i in np.arange(0, num_frames, step=Mem_every)]
    elif Mem_number:
        to_memorize2 = [int(round(i)) for i in np.linspace(0, num_frames, num=Mem_number+2)[:-1]]

        step  = int ((to_memorize2[1] - to_memorize2[0])/4)
        to_memorize = []
        for x in range (1,len(to_memorize2)): # replace by to_memorize2!!
            #print("X", to_memorize[x])
            
            selected_index, motion_blur_value = select_lowest_blur(os.path.join(dataset.image_dir,video),to_memorize2[x],step)
            if motion_blur_value > 100:
            	to_memorize.append(selected_index)

        to_memorize.append(0) # to include the first frame
        to_memorize = sorted(to_memorize) #to sort them (not important)
        print(os.path.join(dataset.image_dir,video))
        print ("Total frames: ", num_frames)
        print ("Selected frames before : ", to_memorize2)
        print ("Selected frames after  : ", to_memorize)

    else:
        raise NotImplementedError


    F_last,M_last = dataset.load_single_image(video,0)
    F_last = F_last.unsqueeze(0)
    M_last = M_last.unsqueeze(0)
    E_last = M_last

    F_last_rev,M_last_rev = dataset.load_single_image_reserse(video,0)
    F_last_rev = F_last_rev.unsqueeze(0)
    M_last_rev = M_last_rev.unsqueeze(0)
    E_last_rev = M_last_rev
    #print("NUM FRAMES:",num_frames)
    for t in range(1,num_frames-1):

        #print('current_frame: {},num_frames: {}, num_objects: {}'.format(t, num_frames, num_objects.numpy()[0]))

        # memorize
        with torch.no_grad():
            prev_key, prev_value = model(F_last[:,:,0], E_last[:,:,0], torch.tensor([num_objects])) 

        if t-1 == 0: # 
            this_keys, this_values = prev_key, prev_value # only prev memory
            #then add the frame of the end of the seq.
            with torch.no_grad():
                prev_key_rev, prev_value_rev = model(F_last_rev[:,:,0], E_last_rev[:,:,0], torch.tensor([num_objects]))

            this_keys = torch.cat([this_keys, prev_key_rev], dim=3)
            this_values = torch.cat([this_values, prev_value_rev], dim=3)

        else:
            this_keys = torch.cat([keys, prev_key], dim=3)
            this_values = torch.cat([values, prev_value], dim=3)
        del prev_key,prev_value

        F_,M_ = dataset.load_single_image(video,t)
        jpg_filename = dataset.load_single_image_name(video, t)

        F_ = F_.unsqueeze(0)
        M_ = M_.unsqueeze(0)
        del M_
        # segment
        with torch.no_grad():
            logit = model(F_[:,:,0], this_keys, this_values, torch.tensor([num_objects]))
        E = F.softmax(logit, dim=1)
        palette = Image.open('/mnt/storage/home/ru20956/scratch/DAVIS/Annotations/480p/bear/00000.png').getpalette()
        test_path = os.path.join('/mnt/storage/home/ru20956/scratch/results', video)
        if not os.path.exists(test_path):
            os.makedirs(test_path)
        if not os.path.exists(test_path.replace("results","logins")):
            os.makedirs(test_path.replace("results","logins"))
        torch.save(E,os.path.join(test_path.replace("results","logins"), jpg_filename+'.pt'))
        del logit
        # update
        if t-1 in to_memorize:
            keys, values = this_keys, this_values
            del this_keys,this_values
        pred = torch.argmax(E[0], dim=0).cpu().numpy().astype(np.uint8)
        E_last = E.unsqueeze(2)
        F_last = F_

        

        img_E = Image.fromarray(pred)
        img_E.putpalette(palette)
        img_E.save(os.path.join(test_path, jpg_filename+'.png'))
        del pred
           
    
    return "DONE"

def evaluate_semisupervised(all_gt_masks, all_res_masks, all_void_masks, metric):
    if all_res_masks.shape[0] > all_gt_masks.shape[0]:
        sys.stdout.write("\nIn your PNG files there is an index higher than the number of objects in the sequence!")
        sys.exit()
    elif all_res_masks.shape[0] < all_gt_masks.shape[0]:
        zero_padding = np.zeros((all_gt_masks.shape[0] - all_res_masks.shape[0], *all_res_masks.shape[1:]))
        all_res_masks = np.concatenate([all_res_masks, zero_padding], axis=0)
    j_metrics_res, f_metrics_res = np.zeros(all_gt_masks.shape[:2]), np.zeros(all_gt_masks.shape[:2])
    for ii in range(all_gt_masks.shape[0]):
        if 'J' in metric:
            j_metrics_res[ii, :] = db_eval_iou(all_gt_masks[ii, ...], all_res_masks[ii, ...], all_void_masks)
        if 'F' in metric:
            f_metrics_res[ii, :] = db_eval_boundary(all_gt_masks[ii, ...], all_res_masks[ii, ...], all_void_masks)
    return j_metrics_res, f_metrics_res

def evaluate(model,Testloader,metric,SET):
    print("ENTER THE EVALUATE, number of test loader is:")
    print(len(Testloader))
    import shutil
    import os
    if os.path.exists('/mnt/storage/home/ru20956/scratch/results'):
        shutil.rmtree('/mnt/storage/home/ru20956/scratch/results')
    
    os.makedirs('/mnt/storage/home/ru20956/scratch/results')
     

    for V in tqdm.tqdm(Testloader):
        num_objects, info = V
        seq_name = info['name']
        #print("SEQ:",seq_name)
        num_frames = info['num_frames']
        start_frame=info['start_frame']
        #print("start frame:",start_frame)
        #print('[{}]: num_frames: {}, num_objects: {}'.format(seq_name, num_frames, num_objects.numpy()[0]))
        #os.rename()
        
        Run_video(Testloader, seq_name, num_frames,start_frame, num_objects,model,Mem_every=None, Mem_number=5) ##  THIS IS UPDATEEEEEEEDDDD
        
    os.system('python /mnt/storage/home/ru20956/scratch/evaluation/evaluation_method.py --task semi-supervised --results_path /mnt/storage/home/ru20956/scratch/results --davis_path /mnt/storage/home/ru20956/scratch/DAVIS_FOR_EVAL --set '+SET)
    

    return "DONE"
        



if __name__ == "__main__":
    torch.set_grad_enabled(False) # Volatile

    def get_arguments():
        parser = argparse.ArgumentParser(description="xxx")
        parser.add_argument("-g", type=str, help="0; 0,1; 0,3; etc", required=True)
        parser.add_argument("-s", type=str, help="set", required=True)
        parser.add_argument("-y", type=int, help="year", required=True)
        parser.add_argument("-D", type=str, help="path to data",default='/smart/haochen/cvpr/data/DAVIS/')
        parser.add_argument("-backbone", type=str, help="backbone ['resnet50', 'resnet18','resnest101']",default='resnet50')
        parser.add_argument("-p", type=str, help="path to weights",default='/smart/haochen/cvpr/weights/davis_youtube_resnet50_799999.pth')
        return parser.parse_args()

    args = get_arguments()

    GPU = args.g
    YEAR = args.y
    SET = args.s
    DATA_ROOT = args.D

    # Model and version
    MODEL = 'STM'
    print(MODEL, ': Testing on DAVIS')

    os.environ['CUDA_VISIBLE_DEVICES'] = GPU
    if torch.cuda.is_available():
        print('using Cuda devices, num:', torch.cuda.device_count())
    

    Testloader = DAVIS_MO_Test(DATA_ROOT, resolution='720p', imset='20{}/{}.txt'.format(YEAR,SET), single_object=(YEAR==16))
    #print("DONE LOADING")
    model = nn.DataParallel(STM(args.backbone))
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    pth = args.p

    model.load_state_dict(torch.load(pth))
    metric = ['J','F']
    
    print(evaluate(model,Testloader,metric,SET))
