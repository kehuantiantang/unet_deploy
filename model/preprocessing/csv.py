import pickle
from collections import defaultdict
import warnings
from multiprocessing.pool import ThreadPool
import cv2
import os
import os.path as osp
import numpy as np
from tqdm import tqdm

from model.preprocessing.logger import self_print as print, Logger

from model.preprocessing.P_R_TP_FP_FN import Polygon_Json
from model.preprocessing.json_polygon import JsonLoader


def get_coor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 变为灰度�?
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  ## 阈值分割得到二值化图片

    contours, heriachy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    final_contours = []
    for i, contour in enumerate(contours):
        # ignore the small polygon
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


def polygon_iou(pred_polygons, gt_polygons, h, w):
    """
    Args:
        pred_polygons: (m, 1, 2)
        gt_polygons: (n, 1, 2)
    Returns:
    """
    m, n = len(pred_polygons), len(gt_polygons)
    pred_masks, gt_masks = [], []
    # path = '/home/jovyan/logs'
    for i, pred_polygon in enumerate(pred_polygons):
        mask = np.zeros(shape=(h, w), dtype=np.uint8)
        # print(pred_polygon.shape, 'pred_polygon')
        cv2.fillPoly(mask, [pred_polygon[:, 0]], 1)
        pred_masks.append(mask.copy())

        # cv2.imwrite(osp.join(path, 'pred_{}.png'.format(i)), mask*255)

    for j, gt_polygon in enumerate(gt_polygons):
        mask = np.zeros(shape=(h, w), dtype=np.uint8)
        # print(gt_polygon.shape, 'gt')
        cv2.fillPoly(mask, [gt_polygon[:, 0]], 1)
        gt_masks.append(mask.copy())

        # cv2.imwrite(osp.join(path, 'gt_{}.png'.format(j)), mask*255)

    m_n = np.zeros(shape=(m, n), dtype=np.float32)
    for i, pred_mask in enumerate(pred_masks):
        for j, gt_mask in enumerate(gt_masks):
            m_n[i][j] = IOU(pred_mask, gt_mask)

    # print(m_n, 'm_n')

    # import sys
    # sys.exit(0)
    return m_n

def pairs_iou(pred_polygons, gt_polygons, h, w, IoU_threshold = 0.5):
    '''
    Args:
        pred_polygons: (m, 1, 2)
        gt_polygons: (n, 1, 2)
        h:
        w:
        IoU_threshold: if IoU score small than IoU threshold it as tp
    Returns:

    '''
    # m, n, computer to obtain iou score
    tp_fp_dict = {'fp_repeat':[], 'tp':[], 'fp':[]}
    try:
        pred_gt_iou_matrix = polygon_iou(pred_polygons, gt_polygons, h, w)
        # m: number of pred polygons, n: number of gt polygons
        m, n = pred_gt_iou_matrix.shape

        if m == 0 or n == 0:
            return tp_fp_dict
        # along the gt axis

        i =  np.argmax(pred_gt_iou_matrix, axis=1)
        # print(i, 'i')
        ious = np.array([pred_gt_iou_matrix[index][v] for index, v in zip(range(len(i)), i)])

        is_select = np.zeros((m, ), dtype=np.bool)

        detected_set =  set()
        # filter the iou score large than IoU threshold
        # print(ious.shape, ious, IoU_threshold)
        overlap_index = np.arange(m)[ious > IoU_threshold]
        overlap_ious = ious[ious > IoU_threshold]
        # high to low order for iou
        indices = np.argsort(-overlap_ious)

        for j in overlap_index[indices]:
            d = i[j]
            if d not in detected_set:
                is_select[j] = True
                detected_set.add(d)

                if len(detected_set) == n:
                    break


        for index in range(len(ious)):
            score = ious[index]
            # print(score, 'score', pred_polygons[0].shape)
            if not is_select[index]:
                if score > IoU_threshold:
                    # repeat
                    tp_fp_dict['fp_repeat'].append((score, pred_polygons[index][:, 0]))
                else:
                    # fp
                    tp_fp_dict['fp'].append((score, pred_polygons[index][:, 0]))
            else:
                # tp
                tp_fp_dict['tp'].append((score, pred_polygons[index][:, 0]))
    except Exception as e:
        warnings.warn('{}'.format(e))

    return tp_fp_dict




def writeJsonAndImg(pj, filename, img_dir, output_dir, extension = 'jpg', write_gt_json = False):
    pj.set_img_path(filename.replace('png', 'jpg'))
    pj.draw_polygons(raw_img_path=img_dir, target_path = output_dir, extension=extension)
    pj.write_json(output_dir, filename.split('.')[0], write_gt_json = write_gt_json)


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

    jl = JsonLoader()

    detect_dir = f'/{output_dir}/{model_name}_mask/'  # yuce
    gt_mask_dir = f'{voc_dir}/SegmentationClass'  # label

    os.makedirs(detect_dir, exist_ok=True)
    os.makedirs(gt_mask_dir, exist_ok=True)

    threadpool = ThreadPool(10)


    polygon_dict = {}
    for filename in tqdm(sorted(os.listdir(pred_mask_dir)), desc='Polygon comp, IoU:%.2f, Method:%s'%(args.IoU, args.compare_method)):
        raw_name, extension = filename.split('.')
        if extension in ['tif', 'jpg', 'png']:
            # find the polygon in each predict images

            polygon_dict[raw_name] = {'tp':[], 'fp':[], 'fp_repeat':[], 'gt':[]}
            pj = Polygon_Json(raw_name)

            pred_img_path = os.path.join(pred_mask_dir, filename)
            pred_img = cv2.imread(pred_img_path)
            h, w = pred_img.shape[:-1]

            pj.set_height_width(h, w)

            # get the polygon from the mask image
            pred_polygons = get_coor(pred_img)


            gt_mask_path = os.path.join(gt_mask_dir, '%s.png'%raw_name)
            if osp.exists(gt_mask_path):
                if args.compare_method == 'mask':
                    gt_mask = cv2.imread(gt_mask_path)
                    # tuple, shape m * n * 2, m object, n points, (x, y)
                    gt_polygons = get_coor(gt_mask)

                elif args.compare_method == 'json':
                    json_path = osp.join(args.input, '%s.json'%raw_name)
                    context = jl.load_json(json_path)
                    gt_polygons = jl.get_objects(context)['polygons']
                    gt_polygons = [np.array(polygon).reshape((-1, 1, 2)) for polygon in gt_polygons]

                else:
                    raise ValueError('Can only use compare method mask or json, but get %s'%args.compare_method)

                # add ground truth to polygon visualization
                pj.gt_polygons = gt_polygons


                # [score1, score2, score3, ...], [polygon1, polygon2, polygon3, ...]
                tp_fp_dict = pairs_iou(pred_polygons, gt_polygons, h, w, args.IoU)

                tps, fps, fp_repeats = tp_fp_dict['tp'], tp_fp_dict['fp'], tp_fp_dict['fp_repeat']

                polygon_dict[raw_name]['tp'], polygon_dict[raw_name]['fp'], polygon_dict[raw_name]['fp_repeat'] = tps, fps, fp_repeats


                polygon_dict[raw_name]['gt'] = [(100, polygon[:, 0]) for polygon in gt_polygons]



                [pj.add_polygon(iou, polygon) for (iou, polygon) in tps]
                [pj.add_polygon(iou, polygon) for (iou, polygon) in fps]
                [pj.add_polygon(iou, polygon) for (iou, polygon) in fp_repeats]

                Logger.debug('tp: %d, fp: %d, fp_repeat: %d, gt:%d, %s'%(len(tps), len(fps), len(fp_repeats), len(gt_polygons), raw_name))


                # save tp/fp/tp_repeat/gt polygon to pkl files
                with open(osp.join(output_dir, 'tp_fp_gt.pkl'), 'wb') as f:
                    pickle.dump(polygon_dict[raw_name], f)
                Logger.info('Save tp/fp/tp_repeat/gt polygon to pkl files, %s'%osp.join(output_dir, 'tp_fp_gt.pkl'))


            else:
                for polygon in pred_polygons:
                    pj.add_polygon(0, polygon[:, 0])

            if status == 'eval':
                kwds = {'pj':pj, 'filename':filename, 'img_dir':f'/{voc_dir}/JPEGImages', 'output_dir':output_dir,
                        'extension':'jpg', 'write_gt_json':args.write_gt_json}
            elif status == 'inference':
                kwds = {'pj':pj, 'filename':filename, 'img_dir':input_dir, 'output_dir':output_dir, 'extension':'tif', 'write_gt_json':args.write_gt_json}
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

