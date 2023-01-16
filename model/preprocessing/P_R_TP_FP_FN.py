import copy
import os
import os.path as osp
import json
import warnings

import cv2
import  numpy as np

from model.preprocessing.logger import Logger


class Polygon_Json(object):

    def __init__(self, name):
        '''
        tools for draw the image (with predict polygon and gt polygon) and json file

        Args:
            name:
        '''
        self.name = name

        self.polygon_dict = {"version":"4.5.7", 'flags':{}, 'shapes':[], "imagePath":-1, "imageData":None,
                             "imageHeight":-1, "imageWidth":-1}
        self.gt_polygons = []
        self.pred_polygons = []
        self.scores = []

    def add_polygons(self, scores, polygons):
        for score, polygon in zip(scores, polygons):
            self.add_polygon(score, polygon)

    def add_polygon(self, score, polygon):
        self.scores.append(score)
        self.pred_polygons.append(polygon)


    def set_height_width(self, height, width):
        '''
        image height and width
        Args:
            height:
            width:

        Returns:

        '''
        self.polygon_dict['imageHeight'] = height
        self.polygon_dict['imageWidth'] = width

    def set_img_path(self, path):
        self.polygon_dict['imagePath'] = path


    def write_json(self, output_dir, name, IoU_threshold = -1, write_gt_json = False):
        '''
        if IoU score small than threshold it will not draw
        Args:
            output_dir:
            name:
            IoU_threshold:

        Returns:

        '''
        polygon_dict = copy.deepcopy(self.polygon_dict)
        for score, polygon in zip(self.scores, self.pred_polygons):
            if score > IoU_threshold:
                # print('%s pred'%name, polygon.shape)
                polygon_template = {"label": '00000000', "score": '%.2f'%(score*100), "points":
                    polygon.tolist(),
                                    "group_id":
                                        "null",
                                    "shape_type": "polygon", "flags": {}}
                polygon_dict["shapes"].append(polygon_template)

        with open(os.path.join(output_dir, '%s.json'%name), 'w', encoding="utf-8") as f:
            json.dump(polygon_dict, f, indent=4)


        if write_gt_json:
            polygon_dict = copy.deepcopy(self.polygon_dict)
            for polygon in self.gt_polygons:
                # print('%s gt'%name, polygon.shape)
                polygon_template = {"label": '00000000', "score": '%.2f'%(100), "points":
                    polygon[:, 0].tolist(),
                                    "group_id":
                                        "null",
                                    "shape_type": "polygon", "flags": {}}
                polygon_dict["shapes"].append(polygon_template)

            with open(os.path.join(output_dir, '%s_gt.json'%name), 'w', encoding="utf-8") as f:
                json.dump(polygon_dict, f, indent=4)


    def draw_polygons(self, raw_img_path, target_path,  extension = 'tif', IoU_threshold = -1):
        '''
        if the iou score small than IoU threshold, it will not draw
        Args:
            raw_img_path:
            target_path:
            extension:
            IoU_threshold:

        Returns:

        '''
        path = osp.join(raw_img_path, '%s.%s'%(self.name, extension))
        try:
            img = cv2.imread(path)
            assert len(img.shape) == 3
            # draw gt
            img = cv2.polylines(img, self.gt_polygons, True, (255, 0, 0), 2)

            # draw pred
            pred_polygons = []
            for score, polygon in zip(self.scores, self.pred_polygons):
                if score > IoU_threshold:
                    pred_polygons.append(polygon)

            img = cv2.polylines(img, pred_polygons, True, (0, 255, 0), 2)
            cv2.imwrite(osp.join(target_path, '%s.jpg'%self.name), img)
        except Exception as e:
            Logger.critical(e)
            warnings.warn("Could not write polygon file into image, %s"%path)


def cal_f1_scoreByIoU(polygon_dict):
    nb_tp, nb_fp, nb_fn, nb_gt = 0, 0, 0, 0
    for image_id, value in polygon_dict.items():
        tps, fps, fp_repeats, gts = value['tp'], value['fp'], value['fp_repeat'], value['gt']
        nb_tp += len(tps)

        nb_fp += len(fps)
        nb_fp += len(fp_repeats)

        nb_gt += len(gts)


    nb_fn = nb_gt - nb_tp
    recall, precision = nb_tp / (nb_tp + nb_fn + 1e-16), nb_tp / (nb_tp + nb_fp + 1e-16)
    f1_score = 2 * precision * recall * 1.0 / (recall + precision + 1e-16)

    acc = nb_tp / (nb_tp + nb_fp + nb_fn + 1e-16)

    info = f'f1_score:{f1_score}, recall:{round(recall, 3)}, prec:{round(precision, 3)}, tp:{nb_tp}, fp:{nb_fp}, fn:{nb_fn}, gt:{nb_gt}'
    Logger.info(info)


    return acc, f1_score

# if __name__ == '__main__':
#     f1_score()

