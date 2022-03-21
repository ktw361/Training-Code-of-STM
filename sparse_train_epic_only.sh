#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --time=1-00:00:00
#SBATCH --mem=16000M
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=ahmad-dk
#SBATCH --mail-type=start,fail,end         # send email if job fails (it could be "end")
#SBATCH --mail-user=ru20956@bristol.ac.uk
# get rid of any modules already loaded
module purge
# load in the module dependencies for this script
module load "languages/anaconda3/2019.07-3.6.5-tflow-1.14"

python sparse_train_epic_only.py -Ddavis  /mnt/storage/home/ru20956/scratch/DAVIS_FOR_EVAL_2fps  -total_iter 100000 -test_iter 10000 -log_iter 500 -batch 32 -backbone resnet50 -save /mnt/storage/home/ru20956/scratch/stm_models/epic_only_weights/ -name epic_0skip_sparse -resume /mnt/storage/home/ru20956/STM_TRAINING/davis_weights/epic_yt_daviss_epic_davis_yt_lr_5_r_2_resnet50_300000_128_229999.pth
#python train_coco.py -Ddavis /mnt/storage/home/ru20956/frtm-vos/val/DAVIS/ -Dcoco /mnt/storage/home/ru20956/scratch/coco/ -backbone resnet50 -save ../coco_weights/


