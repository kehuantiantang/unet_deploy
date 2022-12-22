# coding=utf-8
import re
import warnings

from tqdm import tqdm
import cv2
import os.path as osp
import os

from model.preprocessing.json_polygon import JsonLoader
from model.preprocessing.pascal_voc_utils import Writer
from model.preprocessing.misc import namespace2dict


def read_dir(path):
    '''
    read the file from directory
    :param path:
    :return:
    '''
    files = {}
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('json'):
                name = filename.split('.')[0]
                files[name] = osp.join(root, name)

    return files

def json_to_xml(path, xml_root):
    '''
    load json file and save to xml bounding box
    :param path:
    :param xml_root:
    :return:
    '''
    os.makedirs(xml_root, exist_ok=True)

    jl = JsonLoader()

    for root, _, filenames in os.walk(path):
        for filename in tqdm(filenames, desc='json to xml:'):
            if filename.endswith('json'):
                p = osp.join(root, filename)
                xml_p = osp.join(xml_root, filename.replace('json', 'xml'))

                context = jl.load_json(p)
                objects = jl.get_objects(context)

                if len(objects['bboxes']) > 0:
                    writer = Writer(objects['filename'], objects['width'], objects['height'], database = objects['filename'])
                    writer.addBboxes(objects['bboxes'], objects['category_name'])

                    writer.save(xml_p)

def data_processing(root, target_path):
    # # data path, has *.json, *.jpg file
    # root = '/home/zeta1996/2022_PWDTools/data/labled_0914'
    # # target path to save
    # target_path = '/home/zeta1996/2022_PWDTools/data/tp_0914'

    # json include polygon, point
    json_target = osp.join(target_path, 'json')
    # image gis image
    img_target = osp.join(target_path, 'JPEGImages')
    # segmentation mask
    mask_target = osp.join(target_path, 'SegmentationClass')
    # check whether the polygon and bbox is correctly annotate
    vis_target = osp.join(target_path, 'vis')
    # bbox xml
    # xml_target = osp.join(target_path, 'xml')
    xml_target = osp.join(target_path, 'xml')

    os.makedirs(json_target, exist_ok=True)
    os.makedirs(img_target, exist_ok=True)
    os.makedirs(vis_target, exist_ok=True)
    os.makedirs(mask_target, exist_ok=True)
    os.makedirs(xml_target, exist_ok=True)


    # 获取路径内文件
    file = os.listdir(root)

    for i in tqdm(range(len(file)), desc='data preprocessing'):
        n1 = root + '/' + file[i]
        n2 = re.sub('\(.*?\)', '', n1)
        n3 = n2.replace(" ","")
        os.rename(n1, n3)


    jl = JsonLoader()

    file_path = read_dir(root)
    disease_counter, dis_img_counter, no_dis_img_counter, total_img = 0, 0, 0, 0
    for name, path in tqdm(file_path.items(), desc = 'Data preprocess to folder:'):
        jpg_path = path + '.tif'
        # jpg_path = path + '.tif'
        # gt_path = path + '_gt.jpg'
        json_path = path + '.json'

        try:
            jpg_img = cv2.imread(jpg_path)
            assert len(jpg_img.shape) == 3
            context = jl.load_json(json_path)
            attributes = jl.get_objects(context)
        except Exception as e:
            print(e)
            warnings.warn('Image %s has problem'%jpg_path)
            continue

        nb_disease = len(attributes['polygons'])

        if len(nb_disease) > 0:

            disease_counter += nb_disease
            if nb_disease > 0:
                dis_img_counter += 1
            else:
                no_dis_img_counter += 1
            total_img += 1

            objs = namespace2dict(context)

            # draw bbox image
            # jpg_img_boxes = jl.draw_bboxes(jpg_img, attributes)
            # draw polygon image
            jpg_img_polygons = jl.draw_polygons(jpg_img, attributes)
            # draw image
            jpg_mask = jl.draw_mask(jpg_img, attributes, c=(1, 1, 1), single_channel=True)
            # jpg_mask = cv2.resize(jpg_mask, (512, 512), interpolation=cv2.INTER_NEAREST_EXACT)

            # if nb_disease == 0:
            cv2.imwrite(osp.join(img_target, '%s.jpg'%name), jpg_img)
            # cv2.imwrite(osp.join(vis_target, name + '_%02d.jpg' % nb_disease),
            #             cv2.hconcat([jpg_img_boxes, jpg_img_polygons]))
            cv2.imwrite(osp.join(mask_target, '%s.png'%name), jpg_mask)
            # save_json(osp.join(json_target, name + '_%02d.json' % nb_disease), objs)

    print('The number of disease: %d, disease image/no disease image/total: %d/%d/%d' % (disease_counter,
                                                                                         dis_img_counter,
                                                                                         no_dis_img_counter, total_img))
    print('Convert json to xml bbox', '.' * 50)
    # json_to_xml(json_target, xml_target)

