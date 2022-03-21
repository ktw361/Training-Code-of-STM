#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --time=1-00:00:00
#SBATCH --mem=16000M
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=ahmad-dk
#SBATCH --mail-type=start,fail,end         # send email if job fails (it could be "end")
#SBATCH --mail-user=ru20956@bristol.ac.uk
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
# get rid of any modules already loaded
module purge
# load in the module dependencies for this script
module load "languages/anaconda3/2019.07-3.6.5-tflow-1.14"

python train_epic_only.py -Ddavis  /mnt/storage/home/ru20956/scratch/DAVIS_FOR_EVAL_2fps  -total_iter 100000 -test_iter 4000 -log_iter 500 -batch 128 -backbone resnet50 -save /mnt/storage/home/ru20956/scratch/stm_models/epic_only_weights/ -name 2fps_davis_yt_lr_5_b128_Skip_0_1 -resume /mnt/storage/home/ru20956/STM_TRAINING/davis_weights/davis_youtube_resnet50_799999.pth
#python train_coco.py -Ddavis /mnt/storage/home/ru20956/frtm-vos/val/DAVIS/ -Dcoco /mnt/storage/home/ru20956/scratch/coco/ -backbone resnet50 -save ../coco_weights/


