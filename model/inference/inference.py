import os
import warnings
from datetime import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import sys
sys.path.append('./')
# sys.path.insert(0, '/home/jovyan/model/jbnu')
import argparse
import os.path as osp
from PIL import Image
from tqdm import tqdm
import numpy as np
import cv2
from model.preprocessing.csv import record_coordinate2json
from mmseg.apis import init_segmentor, inference_segmentor
from model.preprocessing.logger import Logger
from model.preprocessing.preprocess import data_processing


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')

    # PWD parameters
    parser.add_argument('--model', default='unet', help='model used [unet, deeplabv3]')
    parser.add_argument('--input',  help='path to input data')
    parser.add_argument('--voc', help='data processing path')
    # parser.add_argument('--output',
    #                     default='/home/zeta1996/job_jsc_ai_2022_serving_mmsegmentation/model/jbnu/unet/polygon/',
    #                     help='path to output model')
    parser.add_argument('--output',  help='path to output model')
    parser.add_argument('--threshold', default='0.5', help='path to output model')
    parser.add_argument('--model_path', help='model used [unet, deeplabv3]')
    parser.add_argument(
        '--gpu_num',
        type=int,
        default=0,
        help='id of gpu to use '
             '(only applicable to non-distributed testing)')
    args = parser.parse_args()

    args.output = '%s_%s'%(args.output, datetime.now().strftime("%Y%m%d_%H%M%S"))
    return args

def read_directory(directory_name, txt_save_path):
    fw = open(txt_save_path, "w", encoding='utf-8')
    for filename in os.listdir(directory_name):
        if filename.split('.')[-1].lower() in ['tif', 'jpg', 'png']:
            filename = filename.split('.')[0]
            fw.write(filename + '\n')

def masktopolygon(im_bg_dir, im_gray_dir, savepath):
    files = {}
    for name in os.listdir(im_bg_dir):
        if name.split('.')[-1].lower() in ['tif', 'jpg', 'png']:
            files[name.split('.')[0]] = osp.join(im_bg_dir, name)


    for name, im_bg_path in tqdm(files.items(), desc='Draw polygon'):

        # im_bg_path, im_gray_path = osp.join(im_bg_dir, '%s.jpg' % name), osp.join(im_gray_dir, '%s.jpg' % name)
        im_gray_path = osp.join(im_gray_dir, '%s.jpg' % name)

        if osp.exists(im_gray_path):
            imbgr, imgray = cv2.imread(im_bg_path), cv2.imread(im_gray_path)
            
            img_gray = cv2.cvtColor(imgray, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(imbgr, contours, -1, (0,0,255), 2)
            
            cv2.imwrite(osp.join(savepath, "%s.jpg"%name), imbgr)
        else:
            warnings.warn("Image is not exist ! %s, %s"%(im_bg_path, im_gray_path))
            # print('ERROR', '*'*100, im_bg_path, im_gray_path)


def main():
    args = parse_args()

    # mmsegmentation
    if args.model == 'unet':
        config_file = '/home/jovyan/model/jbnu/my_model/unet/fcn_unet_s5-d16_128x128_40k_chase_db1.py'
        save_mask_root = f'{args.output}/unet_mask/'
        # config_file = '/home/zeta1996/job_jsc_ai_2022_serving_mmsegmentation/model/jbnu/my_model/unet/fcn_unet_s5-d16_128x128_40k_chase_db1.py'
    elif args.model == 'deeplabv3':
        config_file = '/home/jovyan/model/jbnu/my_model/deeplabv3plus/deeplabv3plus_r50-d8_512x512_20k_voc12aug.py'
        save_mask_root = f'{args.output}/deeplabv3_mask/'
        # config_file = '/home/zeta1996/job_jsc_ai_2022_serving_mmsegmentation/model/jbnu/my_model/deeplabv3plus/deeplabv3plus_r50-d8_512x512_20k_voc12aug.py'
    else:
        config_file, save_mask_root = None, ''
        assert ValueError('The input parameter --model must be unet or deeplabv3, but you use %s'%args.model)


    # read all parameters finish

    target_path = '%s_input'%args.voc


    checkpoint_file = args.model_path
    # print('cuda:'+str(args.gpu_num), '-'*100)
    # model = init_segmentor(config_file, checkpoint_file, device='cuda:'+str(args.gpu_num))
    model = init_segmentor(config_file, checkpoint_file, device='cuda')

    os.makedirs(save_mask_root, exist_ok = True)
    for root, _, filenames in os.walk(args.input):
        for filename in tqdm(filenames, desc='Inference seg:'):
            name, suffix = filename.split('.')
            if suffix.lower() in ['tif', 'jpg', 'jpeg', 'bmp', 'png']:
                img_path = osp.join(root, filename)
                #print('main loop', img_path)
                result = inference_segmentor(model, img_path)[0] #> args.threshold
                img = Image.fromarray(np.uint8(result * 55))

                img.save(osp.join(save_mask_root, '%s.jpg'%name))

    Logger.info('The inference has been finished, and store the predict results into %s'%save_mask_root)

    os.makedirs(args.output, exist_ok=True)
    #ls
    record_coordinate2json(save_mask_root, args.model, args.input, args.output, target_path)
    masktopolygon(args.input, save_mask_root, args.output)


if __name__ == '__main__':
    main()