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
import glob

### My libs
from dataset.dataset_mid import DAVIS_MO_Test
from model.model import STM
from f_boundary import *
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from evaldavis2017.davis2017.davis import DAVIS
from evaldavis2017.davis2017.metrics import db_eval_boundary, db_eval_iou
from evaldavis2017.davis2017 import utils
from evaldavis2017.davis2017.results import Results
from scipy.optimize import linear_sum_assignment

import math
import threading
k = 15
all_iou_per_seq_and_color = {}# to store the scores
all_iou_per_seq_and_color['stm_reconstruction'] = {}
all_iou_per_seq_and_color['forward_backward_matching'] = {}
def Run_video(dataset,video, num_frames,start_frame, num_objects,model,Mem_every=None, Mem_number=None):
    torch.cuda.empty_cache()
    # initialize storage tensors
    if Mem_every:
        to_memorize = [int(i) for i in np.arange(0, num_frames, step=Mem_every)]
    elif Mem_number:
        to_memorize = [int(round(i)) for i in np.linspace(0, num_frames, num=Mem_number+2)[:-1]]
    else:
        raise NotImplementedError
    F_last,M_last = dataset.load_single_image(video,0)
    F_last = F_last.unsqueeze(0)
    M_last = M_last.unsqueeze(0)
    E_last = M_last
    print ('N of zeros', torch.count_nonzero(E_last))
    if torch.count_nonzero(E_last) != 0:
        #print("NUM FRAMES:",num_frames)
        for t in range(1,num_frames):
    
            #print('current_frame: {},num_frames: {}, num_objects: {}'.format(t, num_frames, num_objects.numpy()[0]))
            try:
                # memorize
                with torch.no_grad():
                    prev_key, prev_value = model(F_last[:,:,0], E_last[:,:,0], torch.tensor([num_objects])) 
            except:
                break
    
            if t-1 == 0: # 
                this_keys, this_values = prev_key, prev_value # only prev memory
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
            test_path = os.path.join('/mnt/storage/home/ru20956/scratch/results_mid', video)
            if not os.path.exists(test_path):
                os.makedirs(test_path)
            
            del logit
            # update
            if t-1 in to_memorize:
                keys, values = this_keys, this_values
                del this_keys,this_values
            pred = torch.argmax(E[0], dim=0).cpu().numpy().astype(np.uint8)
            E_last = E.unsqueeze(2)
            F_last = F_
    
            
            if t+1 == num_frames:
                img_E = Image.fromarray(pred)
                img_E.putpalette(palette)
                img_E.save(os.path.join(test_path, jpg_filename+'.png'))  #### there to save the images
                del pred
            import gc
            gc.collect()
           
    
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

def find_jacard_ready(mask1, mask2):
    import numpy as np
    from numpy import asarray
    from PIL import Image
    from sklearn.metrics import jaccard_score



    len_mask = len(np.unique(mask1))
    
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou_score = np.sum(intersection) / np.sum(union)
    
    if (np.sum(union) == 0):
        #print('Image %s => Empty image' % (image1_name.split('/')[-1]))
        #print("Where J=",iou_score)
        return -1

    else:
        #print('IoU is %s' % (iou_score))

        return (np.round(iou_score,3))

def find_f_score_ready(mask1, mask2):
    import numpy as np
    from numpy import asarray
    from PIL import Image

    union = np.logical_or(mask1, mask2)
    ones = np.ones(mask1.shape)
    mask1_m = np.where(mask1 > 0, ones, 0)
    mask2_m = np.where(mask2 > 0, ones, 0)
    #mask1_m = Image.fromarray(mask1_m)
    #mask1_m.show()
    
    if (np.sum(union) == 0):
        print(">>>>>>>>>>>>>> EMPTY IMAGE!!")
        return -1

    else:
        F = db_eval_boundary(mask2_m,mask1_m)
        #print('F is %s' % (F))
        return (np.round(F,3))
    

def stm_propagation_scores(folder1,folder2):
    from numpy import asarray
    for file_path in sorted(glob.glob(folder1 +'/*.png')): # like annotations/predicted/bike-packing/00001.png
        #if ("00000.png" not in file_path and (number_of_files < (number_of_files_including_all_frames - 2))): #exclude the first and last frame
            #ground_truth = file_path.replace("predicted","ground-truth")
            ground_truth = file_path.replace(folder1,folder2)
            print (ground_truth)
            image1 = Image.open(ground_truth)
            image2 = Image.open(file_path)
            mask1 = asarray(image1)
            mask2 = asarray(image2)
            
            for i in np.unique(mask1):
                if i != 0:
                    m =find_jacard_ready(np.where(mask1 == i, mask1, 0),np.where(mask2 == i, mask2, 0))
                    #f= find_f_score_ready(np.where(mask1 == i, mask1, 0),np.where(mask2 == i, mask2, 0))
                    #if m < 0.90:
                    key= ground_truth.split('/')[-2]+'_'+str(i)
                    img_name = ground_truth.split('/')[-1].split('.')[0]
                    if key not in all_iou_per_seq_and_color['stm_reconstruction'].keys():
                        all_iou_per_seq_and_color['stm_reconstruction'][key] = []
                             
                    all_iou_per_seq_and_color['stm_reconstruction'][key].append(m)
                    
                    '''
                    if m < 0.8:
                        print (m)
                        ones = np.ones(mask2.shape) + 244
                        x = np.where(mask2 == i, ones, 0)
                        mm = Image.fromarray(x)
                        mm.show()
                        
                        ones = np.ones(mask1.shape) + 244
                        x = np.where(mask1 == i, ones, 0)
                        mm = Image.fromarray(x)
                        mm.show()
                    '''


def forward_backward_scores(folder1,folder2):
    from numpy import asarray
    for file_path in sorted(glob.glob(folder1 +'/*.png')): # like annotations/predicted/bike-packing/00001.png
        #if ("00000.png" not in file_path and (number_of_files < (number_of_files_including_all_frames - 2))): #exclude the first and last frame
            #ground_truth = file_path.replace("predicted","ground-truth")
            ground_truth = file_path.replace(folder1,folder2)
            print (ground_truth)
            image1 = Image.open(ground_truth)
            image2 = Image.open(file_path)
            mask1 = asarray(image1)
            mask2 = asarray(image2)
            
            for i in np.unique(mask1):
                if i != 0:
                    m =find_jacard_ready(np.where(mask1 == i, mask1, 0),np.where(mask2 == i, mask2, 0))
                    #f= find_f_score_ready(np.where(mask1 == i, mask1, 0),np.where(mask2 == i, mask2, 0))
                    #if m < 0.90:
                    key= ground_truth.split('/')[-2]+'_'+str(i)
                    if key not in all_iou_per_seq_and_color['forward_backward_matching'].keys():
                        all_iou_per_seq_and_color['forward_backward_matching'][key] = []
      
                    all_iou_per_seq_and_color['forward_backward_matching'][key].append(m)
                    
                    '''
                    if m < 0.8:
                        print (m)
                        ones = np.ones(mask2.shape) + 244
                        x = np.where(mask2 == i, ones, 0)
                        mm = Image.fromarray(x)
                        mm.show()
                        
                        ones = np.ones(mask1.shape) + 244
                        x = np.where(mask1 == i, ones, 0)
                        mm = Image.fromarray(x)
                        mm.show()
                    '''

def combine_folder(dirs):
    import glob
    number_of_frames = len(glob.glob(os.path.join(dirs, '*.pt')))
    counter = int (-number_of_frames/2)
    for filename in sorted(glob.glob(os.path.join(dirs, '*.pt'))):
        if os.path.exists(filename.replace("forward","backward")):
            ##print("File: ",filename)
            
            #print(f"=>{counter+1} of {number_of_frames} frames")
            
            E_f = torch.load(filename, map_location=torch.device('cpu'))
            E_b = torch.load(filename.replace("forward","backward"), map_location=torch.device('cpu'))
            linear_f = 1 - counter/(number_of_frames * 2) # 2 is the fraction, less is more range
            linear_b = 1 + counter/(number_of_frames * 2) # 2 is the fraction less is more (e.g. 25%) range
            #x= (counter - number_of_frames/2)/(number_of_frames/2)
            #linear = 1/(1 + math.exp(-20*(x)))
            
            print("Forward factor", round((linear_f),2))
            #print("Backward factor", round((linear_b),2))
            #E_f = E_f * (1-linear)
            #E_b = E_b * (linear)
            #pp = torch.max(E_f[0], dim=0)[0]
            #print("Min", torch.max(pp))
            E = torch.stack([E_f[0] * (linear_f),E_b[0] * (linear_b)])
            #M = E[0].cpu().numpy()
            #print(E[0].shape)
            pred = torch.argmax(E, dim=0).cpu().numpy().astype(np.uint8)
            #print(pred.shape)
            pred_f = E_f[1].cpu().numpy().astype(np.uint8)
            pred_b = E_b[1].cpu().numpy().astype(np.uint8)

            pred = np.where(pred == 0, pred_f, pred_b)
            

            #print(f"Before: PredF > 0 = {np.where(pred_f > 0)[0].size}, Pred >0 = {np.where(pred > 0)[0].size} ")
            
            pred_f[pred_f == 0] = pred_b[pred_f == 0]
            #pred_b[pred_b == 0] = pred_f[pred_b == 0]        
            pred[pred == 0] = pred_f[pred == 0]
            #print (f'unique com {np.unique(pred)} . back {np.unique(pred_b)} , for {np.unique(pred_f)}')
            
            #print(f"After: PredF > 0 = {np.where(pred_f > 0)[0].size}, Pred >0 = {np.where(pred > 0)[0].size} ")


            img_E = Image.fromarray(pred)
            davis_palette = np.repeat(np.expand_dims(np.arange(0, 256), 1), 3, 1).astype(np.uint8)
            davis_palette[:22, :] = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                                         [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                                         [64, 0, 0], [191, 0, 0], [64, 128, 0], [191, 128, 0],
                                         [64, 0, 128], [191, 0, 128], [64, 128, 128], [191, 128, 128],
                                         [0, 64, 0], [128, 64, 0], [0, 191, 0], [128, 191, 0],
                                         [0, 64, 128], [128, 64, 128]]
            img_E.putpalette(davis_palette)
            os.makedirs('/mnt/storage/scratch/ru20956/logits/results/'+"/".join(filename.split('/')[-2:-1]), exist_ok = True)
            img_E.save(os.path.join('/mnt/storage/scratch/ru20956/logits/results/'+"/".join(filename.split('/')[-2:])[:-3]+'.png'))
            counter = counter + 1
            
            import gc
            del E_f
            del E_b
            del E
            del pred_f
            del pred_b
            del pred
            del img_E
            gc.collect()
            

def evaluate(model,Testloader1,Testloader2,metric,SET,dataset_path):
    print("ENTER THE EVALUATE, number of test loader is:")
    print(len(Testloader1))
    import shutil
    import os
    if os.path.exists('/mnt/storage/home/ru20956/scratch/results'):
        shutil.rmtree('/mnt/storage/home/ru20956/scratch/results')
    
    os.makedirs('/mnt/storage/home/ru20956/scratch/results')
     

    for V in tqdm.tqdm(range(len(Testloader1))):
        num_objects, info = Testloader1[V,0] # to get seq name only, 0 just to make the input as tuple (will reutrn seq name only)
        seq_name = info['name']
        print("SEQ:",seq_name)
        dirs = os.path.join('/mnt/storage/scratch/ru20956/logits/forward',seq_name)
        print (dirs + " Started!")
        #generate the png images from the logits
        #if os.path.isdir(dirs) and len(glob.glob(dirs+'/*')) != 0:
        combine_folder(dirs)
        shutil.rmtree(dirs) # to remove the logits
        shutil.rmtree(dirs.replace("forward","backward")) # to remove the logits

        num_objects, info = Testloader1[V] 
        seq_name = info['name']
        ##print("SEQ:",seq_name)

        num_frames = info['num_frames']
        start_frame=info['start_frame']
        ##print("===>>>>>>>start frame:",start_frame)


        #print('[{}]: num_frames: {}, num_objects: {}'.format(seq_name, num_frames, num_objects.numpy()[0]))
        #os.rename()
        #track forward
        Run_video(Testloader1, seq_name, num_frames,start_frame, num_objects,model,Mem_every=None, Mem_number=5) ##  THIS IS UPDATEEEEEEEDDDD
        num_objects, info = Testloader2[V]
        seq_name = info['name']
        #print("SEQ:",seq_name)
        num_frames = info['num_frames']
        start_frame=info['start_frame']
        print("===>>>>>>>start frame:",start_frame)
        #print('[{}]: num_frame

        #track backward
        Run_video(Testloader2, seq_name, num_frames,start_frame, num_objects,model,Mem_every=None, Mem_number=5) ##  THIS IS UPDATEEEEEEEDDDD

        #compute the scores (gt vs reconstructed masks)
        folder1 = os.path.join('/mnt/storage/home/ru20956/scratch/results_mid',seq_name)
        folder2 = os.path.join(os.path.join(dataset_path,'Annotations/480p'),seq_name)
        ##print('folder1:',folder1)
        ##print('folder2:',folder2)
        stm_propagation_scores (folder1,folder2)
        ##shutil.rmtree(folder1) # remove the propagations

        #compute forward_backward similarity

        folder1 = os.path.join('/mnt/storage/scratch/ru20956/results_b',seq_name)
        folder2 = os.path.join('/mnt/storage/scratch/ru20956/results_f',seq_name)

        print('folder1:',folder1)
        print('folder2:',folder2)
        forward_backward_scores (folder1,folder2)
        
        # remove the forward and backward masks
        shutil.rmtree(folder1) 
        shutil.rmtree(folder2)



            

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
    


    #print("DONE LOADING")
    model = nn.DataParallel(STM(args.backbone))
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    pth = args.p

    model.load_state_dict(torch.load(pth))
    metric = ['J','F']
    
    Testloader_b = DAVIS_MO_Test(DATA_ROOT, resolution='480p', imset='20{}/{}.txt'.format(YEAR,SET), single_object=(YEAR==16),direction_m=True)
    Testloader_f = DAVIS_MO_Test(DATA_ROOT, resolution='480p', imset='20{}/{}.txt'.format(YEAR,SET), single_object=(YEAR==16),direction_m=False)
    print(evaluate(model,Testloader_b,Testloader_f,metric,SET,DATA_ROOT))

    import json
    with open('scores.json', 'w') as fp:
        json.dump(all_iou_per_seq_and_color, fp)

