#!/bin/bash

MODEL_NAME=segformer-b5
PROJECT_NAME=${MODEL_NAME}_deploy

#DATASET_NAME=split_test_a
DATASET_NAME=CONTOUR_V3_20221122_145813_R12345_25000_Tag_seg_30

ROOT=/home/khtt/code/insitute_demo/unet_deploy
#ROOT_DATASET=/dataset/khtt/dataset/pine2022/ECOM
ROOT_DATASET=/dataset/khtt/dataset/pine2022/elcom



TRAIN_DATASET=/home/jovyan/datasets/2.labled/${DATASET_NAME}
VOC_DATASET=/home/jovyan/datasets/3.generated/${DATASET_NAME}
OUTPUT_ROOT=/home/jovyan/datasets/5.artifacts/${PROJECT_NAME}_${DATASET_NAME}


docker run -it -u 0 --rm \
	--shm-size 4G \
	-v ${ROOT}/model:/home/jovyan/model \
	-v ${ROOT_DATASET}:/home/jovyan/datasets \
	-v ${ROOT}/logs:/home/jovyan/logs \
    -v /etc/localtime:/etc/localtime:ro \
	--gpus 'device=1' \
 deploy/unet1.0 \
	python /home/jovyan/model/train/train.py --model=${MODEL_NAME}  \
    --input=${TRAIN_DATASET} \
    --voc=${VOC_DATASET} \
    --output=${OUTPUT_ROOT}
