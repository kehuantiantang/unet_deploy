import os
import os.path as osp
import json
import  numpy as np
from tqdm import trange

from model.preprocessing.bbox import make_bbox_label, make_bbox_mask

classes = ['0']
# 初始化二�?数组
result_list = np.array(np.zeros([len(classes), len(classes)]))


def read_label_txt(full_label_name):
    fp = open(full_label_name, mode="r")
    lines = fp.readlines()
    image_width, image_high = 512, 512
    object_list = []
    for line in lines:
        array = line.split()
        x_label_min = (float(array[1]) - float(array[3]) / 2) * image_width
        x_label_max = (float(array[1]) + float(array[3]) / 2) * image_width
        y_label_min = (float(array[2]) - float(array[4]) / 2) * image_high
        y_label_max = (float(array[2]) + float(array[4]) / 2) * image_high
        bbox = [round(x_label_min, 2), round(y_label_min, 2), round(x_label_max, 2), round(y_label_max, 2)]
        category = int(array[0])
        obj_info = {
            'category' : category,
            'bbox' : bbox
        }
        object_list.append(obj_info)
    return object_list


# 计算交集面积
def intersection_area(label_box, detect_box):
    
    x_label_min, y_label_min, x_label_max, y_label_max = label_box
    x_detect_min, y_detect_min, x_detect_max, y_detect_max = detect_box
    if (x_label_max <= x_detect_min or x_detect_max < x_label_min) or ( y_label_max <= y_detect_min or y_detect_max <= y_label_min):
        return 0
    else:
        lens = min(x_label_max, x_detect_max) - max(x_label_min, x_detect_min)
        wide = min(y_label_max, y_detect_max) - max(y_label_min, y_detect_min)
        return lens * wide


# 计算并集面积
def union_area(label_box, detect_box):

    x_label_min, y_label_min, x_label_max, y_label_max = label_box
    x_detect_min, y_detect_min, x_detect_max, y_detect_max = detect_box

    area_label = (x_label_max - x_label_min) * (y_label_max - y_label_min)
    area_detect = (x_detect_max - x_detect_min) * (y_detect_max - y_detect_min)
    inter_area = intersection_area(label_box, detect_box)

    area_union = area_label + area_detect - inter_area

    return area_union


# label 匹配 detect
def label_match_detect(label_list, detect_list):

    #IOU阈值值设�?
    iou_threshold = 0.1

    #true_positive_list:存储识别正确的对象，false_positive_list存储识别错误的对�?
    true_positive_list = []
    false_positive_list = []
        
        
    for detect in detect_list:
        
        detect_bbox = detect['bbox']
        temp_iou = 0.0
            
        for label in label_list:
            label_bbox = label['bbox']
            i_area = intersection_area(label_bbox, detect_bbox)
            u_area = union_area(label_bbox, detect_bbox)

            iou = i_area / u_area

            #只统计最大IOU的预测对�?
            if temp_iou < iou:
                temp_iou = iou

        if temp_iou > iou_threshold:
            true_positive_list.append(detect)
        else:
            false_positive_list.append(detect)
    
    return true_positive_list, false_positive_list, label_list



def f1_score(model_name, input_dir, output_dir):
    '''

    Args:
        model_name: args.model,
        input_dir: args.input,
        output_dir: args.output,

    Returns:
        f1 score
    '''

    dataname = input_dir.split('/')[-1]
    detect_path = f'/home/jovyan/model/jbnu/data/result_{model_name}_{dataname}/' # yuce
    label_path = f'/home/jovyan/model/jbnu/data/label_{dataname}/' # label
    os.makedirs(detect_path, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)
    make_bbox_label('%s_input/SegmentationClass' % input_dir, label_path)
    if model_name== 'unet':
        make_bbox_mask(f'{output_dir}/{model_name}_mask/', detect_path)
    elif model_name== 'deeplabv3':
        make_bbox_mask(f'{output_dir}/{model_name}_mask/', detect_path)
    count_tp = 0
    count_fp = 0
    count_fn = 0
    gt = 0
    count_prec = 0.0
    count_recall = 0.0
    recall = 0
    p = 0
    f1 = 0

    
    all_label = os.listdir(detect_path)

    all_list = []

    for i in trange(len(all_label), desc='TPFP cal:'):
        
        full_detect_path = os.path.join(detect_path, all_label[i])
        # 分离文件名和文件后缀
        label_name, label_extension = os.path.splitext(all_label[i])

        # 拼接标注路径
        full_label_path = os.path.join(label_path, '%s.txt'%label_name)
        full_detect_path = os.path.join(detect_path, '%s.txt'%label_name)
        # 读取标注数据
        if osp.exists(full_label_path) == True and os.path.exists(full_detect_path) == True:
        # if os.path.exists(full_detect_path) == True:
            label_list = read_label_txt(full_label_path)
            # 标注数据匹配detect
            detect_list = read_label_txt(full_detect_path)

            tp_lst, fp_lst, lb_lst = label_match_detect(label_list, detect_list)
            
            obj_info = {
                'label_name' : label_name,
                'tp_lst' : tp_lst,
                'fp_lst' : fp_lst,
                'lb_lst' : lb_lst
            }
            all_list.append(obj_info)
        else:
            if osp.exists(full_label_path) == True:
                myfile = open(full_label_path)
                lines = len(myfile.readlines())
                count_fn += lines
            else:
                count_fn = count_fn



    for lst in all_list:
        tp_lst = lst['tp_lst']
        fp_lst = lst['fp_lst']
        lb_lst = lst['lb_lst']

        count_tp += len(tp_lst)
        count_fp += len(fp_lst)
        if (len(lb_lst)-len(tp_lst)) < 0:
            count_fn += 0
        else:
            count_fn += (len(lb_lst)-len(tp_lst))

        gt += len(lb_lst)
        if count_tp!=0 and count_fp!=0:
            p = count_tp / (count_tp + count_fp) if count_tp + count_fp !=0 else 0

            # recall = count_tp / (count_tp + count_fn)
            recall = count_tp / gt if gt !=0 else 0
            count_prec += p
            count_recall += recall
    count_fn = gt - count_tp

    f1 = 2 * p * recall / (p + recall) if p + recall !=0 else 0
    acc = count_tp / (count_tp + count_fp + count_fn) if (count_tp + count_fp + count_fn) != 0 else 0
    print('tp: ',count_tp,'fp: ',count_fp,'fn: ',count_fn, 'gt:', gt)
    print('pre: ',p,'recall: ',recall)
    print('f1: ',f1, 'acc: ', acc)

    return acc, f1

if __name__ == '__main__':
    f1_score()

