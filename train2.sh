#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --time=4-00:00:00
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

python train_coco.py -Ddavis /mnt/storage/home/ru20956/frtm-vos/val/DAVIS/ -Dcoco /mnt/storage/home/ru20956/scratch/coco/ -backbone resnet50 -save ../coco_weights/
