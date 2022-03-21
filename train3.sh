#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --time=00:12:00
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

python eval.py -g 1 -s val -y 17 -D /mnt/storage/home/ru20956/scratch/DAVIS -p ../coco_weights/coco_res50.pth -backbone resnet50

