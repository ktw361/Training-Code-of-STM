#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --mem=16000M
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=ahmad-dk
#SBATCH --mail-type=fail         # send email if job fails (it could be "end")
#SBATCH --mail-user=ru20956@bristol.ac.uk
# get rid of any modules already loaded
module purge
# load in the module dependencies for this script
module load "languages/anaconda3/2019.07-3.6.5-tflow-1.14"

python eval.py -g '1' -s val -g 1 -y 17 -p /mnt/storage/home/ru20956/STM_TRAINING/davis_weights/epic_2fps_davis_yt_resnet50_100000_32_49999.pth -D /mnt/storage/home/ru20956/scratch/DAVIS

