#!/bin/bash

# DATA
CHECKPOINT_DIR="~/workspace/deep-pose-private/data/checkpoints"
LAST_CKPT="~/workspace/deep-pose-private/data/checkpoints/20200413-000435_2.5e-06_b12_pos0.5_lr300_baseline/e_32"
TRAIN_DATA_DIR="~/datasets/coco2017/train2017"
VAL_DATA_DIR="~/datasets/coco2017/val2017"
TRAIN_ANNOTATIONS="~/datasets/coco2017/annotations/person_keypoints_train2017_single_person_all.json"
VAL_ANNOTATIONS="~/datasets/coco2017/annotations/person_keypoints_val2017_single_person_all.json"

#OTHERS
BATCH=12
LEARNING_RATE=2.5e-4
LR_LOWER_BOUND=2.5e-7
LR_UPPER_BOUND=1e-4
CYCLE_INTERVAL=1500
CYCLE_LR=True
EPOCHS=10
IMAGE_SIZE=256
VALIDATION_INTERVAL=500
WARM_START=1


python ../deep-pose/train.py --batch $BATCH --lr $LEARNING_RATE \
--image_hw $IMAGE_SIZE --val_interval $VALIDATION_INTERVAL \
--ckpt_dir $CHECKPOINT_DIR \
--epochs $EPOCHS \
--cycle_lr $CYCLE_LR \
--lrl $LR_LOWER_BOUND \
--lru $LR_UPPER_BOUND \
--cycle_interval $CYCLE_INTERVAL \
--last_ckpt $LAST_CKPT \
--dtrain $TRAIN_DATA_DIR --dval $VAL_DATA_DIR \
--atrain $TRAIN_ANNOTATIONS --aval $VAL_ANNOTATIONS