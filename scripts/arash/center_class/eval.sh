#!/bin/bash

#cd ../..

# custom config
DATA=../datasets
TRAINER=arash1
SHOTS=16

DATASET=$1
CFG=$2

for SEED in 1 2 3
do
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir output/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/${DATASET}/seed${SEED} \
    --model-dir output/imagenet/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED} \
    --eval-only 
done