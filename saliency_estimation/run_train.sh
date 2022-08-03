#!/bin/bash

# Setting values of arguments

# Traning details
gpus=1
epochs=50
lr=1e-5
num_workers=12
batch_size=48

# Resuming training
ckpt_path=None

# Dataset
val_ratio=0.2
csv_path=../data/baseline.csv
data_path=../data/processed_mri3D


# Logging
seed=None
# DeepGaze1, DeepGaze2, DeepGaze2E, DeepGaze3
model=DeepGaze1

python train.py --gpus $gpus \
    --epochs $epochs \
    --lr $lr \
    --num_workers $num_workers \
    --batch_size $batch_size \
    --ckpt_path $ckpt_path \
    --val_ratio $val_ratio \
    --csv_path $csv_path \
    --data_path $data_path \
    --labels $labels \
    --seed $seed \
    --model $model