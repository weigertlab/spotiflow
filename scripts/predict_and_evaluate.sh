#!/bin/bash

# DATASET=$1

ts -S 1

DATASETS="breastdd_v2 hybiss_spots_v3 merfish_v2 smfish suntag_v2 telomeres terra synaberr_v1"
DATASETS="synthetic_clean synthetic_noisy"

# Predict
for DATASET in $DATASETS; do
    ts --gpus=1 python predict_dataset.py --data-dir /data/spots/datasets/${DATASET} --model-dir "/data/spotiflow_v2_experiments/lightning/${DATASET}_unet_lvs4_fmaps32_direct_convperlv3_lossbce_epochs200_lr00003_bsize4_ksize3_sigma1_crop512_posweight10_seed42_tormenter" --out-dir /data/spots/results/${DATASET}/spotiflow_1lv_preds_lightning --which last --opt-split val
done;


# Evaluate
for DATASET in $DATASETS; do
   ts python retrieve_metrics.py --ground-truth /data/spots/datasets/${DATASET}/test --predictions /data/spots/results/${DATASET}/spotiflow_1lv_preds_lightning/test --display-cutoff 3 --cutoffs 1 5 .25 --outfile /home/adomi/Workspace/spot-detector-experiments/evaluation/metrics_v2/${DATASET}/spotiflow_lightning.csv
done;

# # COMPILATION

# DATASETS="breastdd_v2 hybiss_spots_v3 merfish_v2 smfish suntag_v2 telomeres terra synaberr_v1 synthetic_clean synthetic_noisy"
# MODEL="breast|hybiss|merfish|smfish|suntag|synaberr|telomeres|terra"

# # Predict
# for DATASET in $DATASETS; do
#     ts --gpus=1 python predict_dataset.py --data-dir /data/spots/datasets/${DATASET} --model-dir /data/spotiflow_v2_experiments/compilations/lightning/${MODEL}_unet_lvs4_fmaps32_direct_convperlv3_lossbce_epochs200_lr00003_bsize4_ksize3_sigma1_crop512_posweight10_seed42_tormenter --out-dir /data/spots/results/${DATASET}/spotiflow_1lv_preds_all_lightning --which last #  --opt-split val
# done;

# # Evaluate
# for DATASET in $DATASETS; do
#     ts python retrieve_metrics.py --ground-truth /data/spots/datasets/${DATASET}/test --predictions /data/spots/results/${DATASET}/spotiflow_1lv_preds_all_lightning/test --display-cutoff 3 --cutoffs 1 5 .25 --outfile /home/adomi/Workspace/spot-detector-experiments/evaluation/metrics_v2/${DATASET}/spotiflow_all_lightning.csv
# done;