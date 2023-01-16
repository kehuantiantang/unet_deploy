#!/bin/bash

MODEL_NAME=segformer-b5
PROJECT_NAME=${MODEL_NAME}_deploy

DATASET_NAME=CONTOUR_V3_20221122_145813_R12345_25000_Tag_seg_30

#DATASET_NAME=small

ROOT_DATASET=/dataset/khtt/dataset/pine2022/elcom
#ROOT_DATASET=/dataset/khtt/dataset/pine2022/ECOM/

ROOT=/home/khtt/code/insitute_demo/unet_deploy


EVAL_DATASET=/home/jovyan/datasets/2.labled/${DATASET_NAME}
VOC_DATASET=/home/jovyan/datasets/3.generated/${DATASET_NAME}
OUTPUT_ROOT=/home/jovyan/datasets/7.evaluations/${PROJECT_NAME}_${DATASET_NAME}


docker run -it -u 0 --rm \
	--shm-size 4G \
	-v ${ROOT}/model:/home/jovyan/model \
	-v ${ROOT_DATASET}:/home/jovyan/datasets \
	-v ${ROOT}/logs:/home/jovyan/logs \
  -v /etc/localtime:/etc/localtime:ro \
    --gpus 'device=1' \
 deploy/unet1.0 \
    python /home/jovyan/model/eval/eval.py --model=${MODEL_NAME} \
    --input=${EVAL_DATASET} \
    --voc=${VOC_DATASET} \
    --output=${OUTPUT_ROOT} \
    --IoU 0.5 \
    --eval=mIoU \
    --model_path=/home/jovyan/datasets/5.artifacts/segformer-b5_deploy_CONTOUR_V3_20221122_145813_R12345_25000_Tag_70+7000-20230112_105612/epoch_100.pth \
    --compare_method=mask

