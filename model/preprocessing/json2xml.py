# coding=utf-8

import sys

from json_polygon import JsonLoader
from pascal_voc_utils import Reader, Writer

sys.path.append('/')
sys.path.append('../jbnu/')
import os
import os.path as osp

from collections import defaultdict

import shutil
from tqdm import tqdm


# only modify this, 202010 pine validation data
# ==================================================================================================================================


def get_object_finalValidation20201113(context):
    objects = {'filename':context.filename, 'width':context.img_size.width, 'height':context.img_size.height}
    bboxes, category_name = [], []
    for bbox, data in zip(context.bboxes, context.data):
        bboxes.append([int(b) for b in bbox])
        if data.status != '':
            category_name.append('dt')
        elif data.level in ['middle', 'high']:
            category_name.append('disease')
        elif data.level in ['low']:
            category_name.append('disease_am')


    objects['bboxes'], objects['category_name']  = bboxes, category_name
    return objects
replace_pair_finalValidation20201113 = [('고사상태', 'status'), ('구분등급', 'level'), ('중', 'middle'), ('상', 'high'), ('하', 'low')]
# ================================================================================================================


def get_object_finalValidation20201030(context):
    objects = {'filename':context.filename, 'width':context.img_size.width, 'height':context.img_size.height,
               'bboxes':[]}
    bboxes, category_name = [], []
    if 'bboxes' in context.__dict__:
        for bbox, data in zip(context.bboxes, context.data):
            bboxes.append([int(b) for b in bbox])
            if data.comment == 'disease' or data.comment == 'disease_am':
                status = data.status
                if data.comment == 'unknown':
                    status = 'low'
                category_name.append('disease|' + status)

        objects['bboxes'], objects['category_name']  = bboxes, category_name
    return objects
replace_pair_finalValidation20201030 = [('고사상태', 'status'), ('구분등급', 'level'), ('중', 'middle'), ('상', 'high'), ('하',
                                                                                                                'low'), ('사람', 'low'), ('AI', 'low'), ('unknow', 'unknown')]




# ================================================================================================================
def get_object_4_dataset(context):
    objects = {'filename':context.filename, 'width':context.img_size.width, 'height':context.img_size.height}
    bboxes, category_name = [], []
    for bbox, data in zip(context.bboxes, context.data):
        bboxes.append([int(b) for b in bbox])
        if  str(data.Class).strip() == '4':
            category_name.append('dt')
        elif str(data.Class).strip() == '3':
            category_name.append('fp')
        elif str(data.Class).strip() == '2':
            category_name.append('unknown')
        elif str(data.Class).strip() == '1':
            category_name.append('disease')

    objects['bboxes'], objects['category_name']  = bboxes, category_name
    return objects
replace_pair_4_dataset = []
# ================================================================================================================

def filter_xml(src, target):
    '''
    filter xml and jpg file, if xml has bbox, copy xml and jpg to target folder
    :param src:
    :param target:
    :return:
    '''
    files = defaultdict(set)
    for root, _, filenames in os.walk(src):
        for filename in filenames:
            if filename.endswith('xml') or filename.endswith('jpg'):
                basename = filename.split('.')[0]
                p = osp.join(root, filename)

                if filename.endswith('xml'):
                    if len(Reader(p).get_objects()['bboxes']) > 0:
                        files[basename].append(p)
                else:
                    files[basename].append(p)

    os.makedirs(target, exist_ok=True)
    for key, value in files.items():
        if len(value) == 2:
            for v in value:
                shutil.copy(v, target)
        elif len(value) > 2:
            print('large', key, value)


path = '/dataset/khtt/dataset/multi-resPine/all_files'
xml_root = '/dataset/khtt/dataset/multi-resPine/disease_level'
os.makedirs(xml_root, exist_ok=True)

# jl = JsonLoader(get_object_finalValidation20201030)
# jl = JsonLoader(get_object_4_dataset)
jl = JsonLoader()

for root, _, filenames in os.walk(path):
    for filename in tqdm(filenames):
        if filename.endswith('json'): #and 'disease_patches' in root:
            p = osp.join(root, filename)
            xml_p = osp.join(xml_root, filename.replace('json', 'xml'))

            context = jl.load_json(p, replace_pair_finalValidation20201030)
            objects = jl.get_objects(context)

            if len(objects['bboxes']) > 0:
                writer = Writer(objects['filename'], objects['width'], objects['height'], database = objects['filename'])
                writer.addBboxes(objects['bboxes'], objects['category_name'])

                writer.save(xml_p)





