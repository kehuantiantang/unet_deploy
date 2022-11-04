import json
import cv2
import os
import os.path as osp
import numpy as np



def get_coor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 变为灰度�?
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  ## 阈值分割得到二值化图片

    contours, heriachy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    final_contours = []
    for i, contour in enumerate(contours):
        if len(contour) < 3 or cv2.contourArea(contour) < 20:
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

                if iou > 0.01:
                    has_overlap = True
                    out_polygon.append(pred_polygon[:, 0])
                    score_record.append(iou)
                    gt_index[index] = 1
                    break

        if not has_overlap:
            out_polygon.append(pred_polygon[:, 0])
            score_record.append(0)

    return score_record, out_polygon



class Polygon_Json(object):
    def __init__(self):
        self.polygon_dict = {"version":"4.5.7", 'flags':{}, 'shapes':[], "imagePath":-1, "imageData":None,
                        "imageHeight":-1, "imageWidth":-1}

    def add_polygons(self, scores, polygons):
        for score, polygon in zip(scores, polygons):
            self.add_polygon(score, polygon)

    def add_polygon(self, score, polygon):
        polygon_template = {"label": '01110200', "score": '%.2f'%(score*100), "points":
            polygon.tolist(),
                            "group_id":
            "null",
                            "shape_type": "polygon", "flags": {}}
        self.polygon_dict["shapes"].append(polygon_template)

    def set_height_width(self, height, width):
        self.polygon_dict['imageHeight'] = height
        self.polygon_dict['imageWidth'] = width

    def set_path(self, path):
        self.polygon_dict['imagePath'] = path


    def write_json(self, output_dir, name):
        with open(os.path.join(output_dir, '%s.json'%name), 'w', encoding="utf-8") as f:
            json.dump(self.polygon_dict, f, indent=4)


def record_coordinate2json(pred_mask_dir, model_name, y, output_dir, voc_dir):

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

    for filename in os.listdir(pred_mask_dir):
        if filename.split('.')[-1].lower() in ['tif', 'jpg', 'png']:
            pj = Polygon_Json()

            # print('polygon 86 ', filename, '!'*10)

            pred_img_path = os.path.join(pred_mask_dir, filename)
            pred_img = cv2.imread(pred_img_path)
            h, w = pred_img.shape[:-1]

            pj.set_height_width(h, w)

            pred_polygons = get_coor(pred_img)

            gt_mask_path = os.path.join(gt_mask_dir, filename.split('.')[0] + '.png')
            # print('polygon path 94 ', gt_mask_path, '!'*10)


            if osp.exists(gt_mask_path):
                gt_mask = cv2.imread(gt_mask_path)
                gt_polygons = get_coor(gt_mask)

                # print('gt,pred size ', gt_mask.shape, pred_img.shape)
                # print('polygon  100 gt ', len(gt_polygons), gt_polygons[0].shape if len(gt_polygons) > 0 else -1,
                #       '!'*10)
                # print('polygon  101 pred ', len(pred_polygons), pred_polygons[0].shape if len(pred_polygons) > 0 else
                # -1,
                #       '!'*10)

                part_ious, with_score_polygons = pairs_iou(pred_polygons, gt_polygons, h, w)

                # print('polygon  104 overlap ', len(with_score_polygons), with_score_polygons[0].shape if len(
                #     with_score_polygons) > 0 else -1, '!'*10)

                pj.add_polygons(part_ious, with_score_polygons)

            else:
                for polygon in pred_polygons:
                    pj.add_polygon(0, polygon[:, 0])

            pj.set_path(filename.replace('png', 'jpg'))
            pj.write_json(output_dir, filename.split('.')[0])
