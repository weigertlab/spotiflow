#!/bin/bash

DATASET=$1

# Predict
python predict_dataset.py --data-dir /data/spots/datasets/${DATASET}/ --model-dir /data/spotipy_torch_v2_experiments/${DATASET}_unet_direct_relu_4lv/ --out-dir /data/spots/results/${DATASET}/spotipy_torch_preds --opt-split val --which last

# Evaluate
python retrieve_metrics.py --ground-truth /data/spots/datasets/${DATASET}/test --predictions /data/spots/results/${DATASET}/spotipy_torch_preds/test --display-cutoff 3 --cutoffs 1 5 .25