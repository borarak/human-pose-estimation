#!/bin/bash

# DATA
# TODO: Remove checkpoint dir redundancy
CHECKPOINT_DIR="/home/raktim/checkpoints/"
LAST_CKPT="/home/raktim/checkpoints/e_19_i7200_g111780"
TRAIN_DATA_DIR="/home/raktim/datasets/coco2017/train2017"
VAL_DATA_DIR="/home/raktim/datasets/coco2017/val2017"
TRAIN_ANNOTATIONS="/home/raktim/datasets/coco2017/annotations/person_keypoints_train2017_single_person_all.json"
VAL_ANNOTATIONS="/home/raktim/datasets/coco2017/annotations/person_keypoints_val2017_single_person_all.json"

#OTHERS
BATCH=6
LEARNING_RATE=1e-4
LR_LOWER_BOUND=5.5e-5
LR_UPPER_BOUND=5.5e-4
CYCLE_INTERVAL=3500
CYCLE_LR=0
EPOCHS=100
IMAGE_SIZE=256
VALIDATION_INTERVAL=500
WARM_START=1
MODEL_NAME='GatedHGNet'
STACKS=4


python ../deep-pose/train.py --batch $BATCH --lr $LEARNING_RATE \
--model_name $MODEL_NAME \
--image_hw $IMAGE_SIZE --val_interval $VALIDATION_INTERVAL \
--ckpt_dir $CHECKPOINT_DIR \
--epochs $EPOCHS \
--stacks $STACKS \
--warm_start $WARM_START \
--cycle_lr $CYCLE_LR \
--lrl $LR_LOWER_BOUND \
--lru $LR_UPPER_BOUND \
--cycle_interval $CYCLE_INTERVAL \
--last_ckpt $LAST_CKPT \
--dtrain $TRAIN_DATA_DIR --dval $VAL_DATA_DIR \
--atrain $TRAIN_ANNOTATIONS --aval $VAL_ANNOTATIONS