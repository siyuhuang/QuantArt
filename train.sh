# # Stage-1
# python -u main.py \
# --base configs/vqgan_wikiart.yaml \
# -t True --gpus 0,1,2,3

# Stage-2
python -u main.py \
--base configs/coco2art.yaml \
-t True --gpus 0,