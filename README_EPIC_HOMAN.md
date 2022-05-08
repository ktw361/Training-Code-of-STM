```python
python eval_custome_light.py \
    -g '0' -s val -y 17 \
    -p ../davis_weights/epic_dense_epic_0skip_sparse_resnet50_100000_32_29999.pth \
    -D ~/datasets/EPIC_DAVIS \
    --external ~/data/epic/rgb_root \
    --calc_num
```


epic_frame_predict.py:
```python
python epic_frame_predict.py --ann_file ~/data/epic_analysis/clean_tiny_gt.txt --save_dir ~/data/epic_analysis/interpolation/
```
