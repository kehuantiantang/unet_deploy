#!/bin/bash

MODEL_NAME=unet
PROJECT_NAME=${MODEL_NAME}_deploy

DATASET_NAME=small
#DATASET_NAME=final_small

#DATASET_NAME=CONTOUR_V3_20221122_145813_R12345_25000_Tag_orderedAll_det30
#DATASET_NAME=CONTOUR_V3_20221122_145813_R12345_25000_Tag_30
#DATASET_NAME=CONTOUR_V3_20221122_145813_R12345_25000_Tag_orderedAll

DATASET_NAME=CONTOUR_V3_20221122_145813_R12345_25000_Tag_seg_30


ROOT_DATASET=/dataset/khtt/dataset/pine2022/elcom
#ROOT_DATASET=/dataset/khtt/dataset/pine2022/ECOM/

ROOT=/home/khtt/code/insitute_demo/${PROJECT_NAME}


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
    --model_path=/home/jovyan/model/weights/${MODEL_NAME}/best_mIoU_iter_12000unet.pth \
    --compare_method=mask

