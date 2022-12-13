#!/bin/bash

MODEL_NAME=unet
PROJECT_NAME=${MODEL_NAME}_deploy

DATASET_NAME=CONTOUR_V3_20221122_230805_DIV30_DEIDF

ROOT=/mnt/d/jbu/평가프로그램_v2_20221107/${PROJECT_NAME}
ROOT_DATASET=/mnt/d/jbu/0.datasets


INFERENCE_DATASET=/home/jovyan/datasets/2.labeled/${DATASET_NAME}
VOC_DATASET=/home/jovyan/datasets/3.generated/${DATASET_NAME}
OUTPUT_ROOT=/home/jovyan/datasets/4.detected/${PROJECT_NAME}_${DATASET_NAME}



docker run -it -u 0 --rm \
	--shm-size 4G \
	-v ${ROOT}/model:/home/jovyan/model \
	-v ${ROOT_DATASET}:/home/jovyan/datasets \
	-v ${ROOT}/logs:/home/jovyan/logs \
  -v /etc/localtime:/etc/localtime:ro \
    --gpus all \
 deploy/unet1.0 \
    python /home/jovyan/model/inference/inference.py --model=${MODEL_NAME} \
    --input=${INFERENCE_DATASET} \
    --voc=${VOC_DATASET} \
    --output=${OUTPUT_ROOT} \
 --model_path=/home/jovyan/model/weights/${MODEL_NAME}/best_mIoU_iter_12000unet.pth