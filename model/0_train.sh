#!/bin/bash

MODEL_NAME=unet
PROJECT_NAME=${MODEL_NAME}_deploy

DATASET_NAME=20220929_abcdefg
ROOT_DATASET=/dataset/khtt/dataset/pine2022/ECOM
ROOT=/home/khtt/code/insitute_demo/${PROJECT_NAME}


TRAIN_DATASET=/home/jovyan/datasets/2.labled/${DATASET_NAME}
VOC_DATASET=/home/jovyan/datasets/3.generated/${DATASET_NAME}
OUTPUT_ROOT=/home/jovyan/datasets/5.artifacts/${PROJECT_NAME}_${DATASET_NAME}


docker run -it -u 0 --rm \
	--shm-size 4G \
	-v ${ROOT}/model:/home/jovyan/model \
	-v ${ROOT_DATASET}:/home/jovyan/datasets \
	-v ${ROOT}/logs:/home/jovyan/logs \
    -v /etc/localtime:/etc/localtime:ro \
	--gpus all \
 deploy/unet1.0 \
	python /home/jovyan/model/train/train.py --model=${MODEL_NAME}  \
    --input=${TRAIN_DATASET} \
    --voc=${VOC_DATASET} \
    --output=${OUTPUT_ROOT}
