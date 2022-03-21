#!/usr/bin/env bash
srun \
    --partition gpu \
    --gres gpu:1 \
    --mem 16GB \
    -t 0-00:10 \
    python eval.py -g 1 -s val -y 17 -D /mnt/storage/home/ru20956/frtm-vos/val/DAVIS -p davis_youtube_resnet50_799999_170.pth  -backbone resnet50
