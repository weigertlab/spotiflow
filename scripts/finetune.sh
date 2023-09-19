#!/bin/bash

PRETRAINED_DATASET="synaberr"
FINETUNE_DATASET="hybiss_v4"
FINETUNE_DATASET_LONG="hybiss_spots_v4"
N_FILES="1 3 5 10 50 100"
ts -S 2

# Train
# for CURR_N_FILES in $N_FILES; do
#     ts --gpus=1 python train.py --config ../configs/finetune_${PRETRAINED_DATASET}_to_${FINETUNE_DATASET}.yaml --logger wandb --save-dir /data/spotipy_torch_v2_experiments/lightning/final_flow/finetune_${CURR_N_FILES}files --flow --batch-norm --skip-bg-remover --mode slim --max-files $CURR_N_FILES
# done

# Predict

# for CURR_N_FILES in $N_FILES; do
#     ts --gpus 1 python predict_dataset.py --data-dir /data/spots/datasets/${FINETUNE_DATASET_LONG} --model-dir "/data/spotipy_torch_v2_experiments/lightning/final_flow/finetune_${CURR_N_FILES}files/${FINETUNE_DATASET_LONG}_unet_lvs4_fmaps32_slim_convperlv3_lossbce_epochs30_lr0_0003_bsize4_ksize3_sigma1_crop512_posweight10_seed42_mode_slim_flow_True_bn_True_aug0_5_tormenter_skipbgremover_finetuned" --out-dir /data/spots/results/${FINETUNE_DATASET_LONG}/spotipy_torch_all_preds_lightning_final_flow/finetune_${CURR_N_FILES}files --which best #  --opt-split val
# done

# Evaluate
for CURR_N_FILES in $N_FILES; do
    ts python retrieve_metrics.py --ground-truth /data/spots/datasets/${FINETUNE_DATASET_LONG}/test --predictions /data/spots/results/${FINETUNE_DATASET_LONG}/spotipy_torch_all_preds_lightning_final_flow/finetune_${CURR_N_FILES}files/test --display-cutoff 3 --cutoffs 1 5 .25
done