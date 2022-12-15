import json
import warnings
from collections import defaultdict
from multiprocessing.pool import ThreadPool

import cv2
import os
import os.path as osp
import numpy as np
from tqdm import tqdm

from model.preprocessing.P_R_TP_FP_FN import Polygon_Json


def get_coor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 变为灰度�?
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  ## 阈值分割得到二值化图片

    contours, heriachy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    final_contours = []
    for i, contour in enumerate(contours):
        if len(contour) < 20 or cv2.contourArea(contour) < 16*16:
            continue

        final_contours.append(contour)

    # list
    return final_contours

def IOU(pred, label):
    pred = pred.flatten()
    label = label.flatten()

    inter = np.sum(pred & label)
    union = np.sum(pred | label)
    if union == 0:
        return 0
    else:
        return inter / union

def pairs_iou(pred_polygons, gt_polygons, h, w):
    """

    Args:
        pred_polygons: (n_obj, n_point, 1, 2)
        gt_polygons: (n_obj, n_point, 1, 2)
        h:
        w:

    Returns:

    """
    # pred_index = [0 for _ in range(len(pred_polygons))]
    gt_index = [0 for _ in range(len(gt_polygons))]

    out_polygon, score_record =  [], []
    for pred_polygon in pred_polygons:
        # obj (n_point, 1, 2)
        pred_mask = np.zeros(shape=(h, w), dtype=np.uint8)
        # print(pred_polygon[:, 0])
        cv2.fillPoly(pred_mask, [pred_polygon[:, 0]], 1)


        has_overlap = False
        for index, gt_polygon in enumerate(gt_polygons):

            if gt_index[index] == 0:
                gt_mask = np.zeros(shape=(h, w), dtype=np.uint8)
                cv2.fillPoly(gt_mask, [gt_polygon[:, 0]], 1)

                iou = IOU(pred_mask, gt_mask)

                if iou > 0.001:
                    has_overlap = True
                    out_polygon.append(pred_polygon[:, 0])
                    score_record.append(iou)
                    gt_index[index] = 1
                    break

        if not has_overlap:
            out_polygon.append(pred_polygon[:, 0])
            score_record.append(0)

    return score_record, out_polygon




def writeJsonAndImg(pj, filename, img_dir, output_dir, IoU_threshold = -1, extension = 'jpg'):
    pj.set_img_path(filename.replace('png', 'jpg'))
    pj.draw_polygons(raw_img_path=img_dir, target_path = output_dir, extension=extension, IoU_threshold = IoU_threshold)
    pj.write_json(output_dir, filename.split('.')[0], IoU_threshold = IoU_threshold)


def record_coordinate2json(pred_mask_dir, args, voc_dir, status):
    '''
    generate the json file
    Args:
        pred_mask_dir:
        model_name:
        output_dir: output json, image file dir
        voc_dir: mask dir

    Returns:
    '''
    model_name, input_dir, output_dir = args.model, args.input, args.output

    if model_name == 'unet':
        detect_dir = f'/{output_dir}/{model_name}_mask/'  # yuce
        gt_mask_dir = f'{voc_dir}/SegmentationClass'  # label
    elif model_name == 'deeplabv3':
        detect_dir = f'/{output_dir}/{model_name}_mask/'  # yuce
        gt_mask_dir = f'{voc_dir}/SegmentationClass'  # label
    else:
        print('Can not use the model')

    os.makedirs(detect_dir, exist_ok=True)
    os.makedirs(gt_mask_dir, exist_ok=True)

    threadpool = ThreadPool(10)


    polygon_dict = defaultdict(lambda :{'gt':[], 'pred':[], 'pred_score':[]})
    for filename in tqdm(sorted(os.listdir(pred_mask_dir)), desc='Polygon compare'):
        raw_name, extension = filename.split('.')
        if extension in ['tif', 'jpg', 'png']:
            # find the polygon in each predict images

            pj = Polygon_Json(raw_name)

            pred_img_path = os.path.join(pred_mask_dir, filename)
            pred_img = cv2.imread(pred_img_path)
            h, w = pred_img.shape[:-1]

            pj.set_height_width(h, w)

            # obtain the predict polygons
            pred_polygons = get_coor(pred_img)
            # print("153, polygon length", len(pred_polygons), 'already generate the predict polygon', pred_polygons[
            #     0].shape)


            gt_mask_path = os.path.join(gt_mask_dir, '%s.png'%raw_name)
            if osp.exists(gt_mask_path):
                gt_mask = cv2.imread(gt_mask_path)
                # shape m * n * 2, m object, n points, (x, y)
                gt_polygons = get_coor(gt_mask)

                pj.gt_polygons = gt_polygons
                polygon_dict[raw_name]['gt'] = gt_polygons

                # print("162, polygon length", len(gt_polygons), 'already generate the gt polygon',
                #       gt_polygons[0].shape)


                # [score1, score2, score3, ...], [polygon1, polygon2, polygon3, ...]
                part_ious, with_polygons = pairs_iou(pred_polygons, gt_polygons, h, w)

                # print('overlap polygon:', len(with_polygons), with_polygons[0].shape, len(part_ious), len(pred_polygons))
                # print('iou score ',part_ious)

                polygon_dict[raw_name]['pred'] = with_polygons
                polygon_dict[raw_name]['pred_score'] = part_ious


                pj.add_polygons(part_ious, with_polygons)

            else:
                for polygon in pred_polygons:
                    pj.add_polygon(0, polygon[:, 0])

            if status == 'eval':
                kwds = {'pj':pj, 'filename':filename, 'img_dir':f'/{voc_dir}/JPEGImages', 'output_dir':output_dir,
                        'extension':'jpg'}
            elif status == 'inference':
                kwds = {'pj':pj, 'filename':filename, 'img_dir':input_dir, 'output_dir':output_dir, 'extension':'tif'}
            else:
                raise ValueError('Status must be [eval, inference] but obtain %s'%status)

            threadpool.apply_async(writeJsonAndImg, kwds = kwds)
            # writeJsonAndImg(**kwds)


            # pj.set_img_path(filename.replace('png', 'jpg'))
            # pj.draw_polygons(raw_img_path=f'/{voc_dir}/JPEGImages', target_path = output_dir, extension='jpg')
            # pj.write_json(output_dir, filename.split('.')[0])

    threadpool.close()
    threadpool.join()

    return polygon_dict

