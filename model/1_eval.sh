#!/bin/bash

MODEL_NAME=unet
PROJECT_NAME=${MODEL_NAME}_deploy

DATASET_NAME=CONTOUR_V3_20221122_230805_DIV30_DEIDF
#DATASET_NAME=test

ROOT=/mnt/d/jbu/평가프로그램_v3/${PROJECT_NAME}
ROOT_DATASET=/mnt/d/jbu/0.datasets


EVAL_DATASET=/home/jovyan/datasets/2.labeled/${DATASET_NAME}
VOC_DATASET=/home/jovyan/datasets/3.generated/${DATASET_NAME}
OUTPUT_ROOT=/home/jovyan/datasets/7.evaluations/${PROJECT_NAME}_${DATASET_NAME}


docker run -it -u 0 --rm \
	--shm-size 4G \
	-v ${ROOT}/model:/home/jovyan/model \
	-v ${ROOT_DATASET}:/home/jovyan/datasets \
	-v ${ROOT}/logs:/home/jovyan/logs \
  -v /etc/localtime:/etc/localtime:ro \
    --gpus all \
 deploy/unet1.0 \
    python /home/jovyan/model/eval/eval.py --model=${MODEL_NAME} \
    --input=${EVAL_DATASET} \
    --voc=${VOC_DATASET} \
    --output=${OUTPUT_ROOT} \
    --IoU=0.3 \
    --eval=mIoU \
    --model_path=/home/jovyan/model/weights/${MODEL_NAME}/best_mIoU_iter_12000unet.pth

