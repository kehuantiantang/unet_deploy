# Copyright (c) OpenMMLab. All rights reserved.
# encoding='utf-8'

import sys
sys.path.append('./')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from model.preprocessing.logger import Logger, VERSION
from model.preprocessing.logger import self_print as print

import pprint
import json
from datetime import datetime
import os
from model.preprocessing.P_R_TP_FP_FN import cal_f1_scoreByIoU
from model.preprocessing.csv import record_coordinate2json
import shutil
import time
import warnings
import mmcv
import torch
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction
from mmseg import digit_version
from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import build_ddp, build_dp, get_device
from mmseg.utils import setup_multi_processes

import argparse
import os.path as osp
from PIL import Image
from tqdm import tqdm

import numpy as np
import cv2

from mmseg.apis import init_segmentor, inference_segmentor
from model.preprocessing.preprocess import data_processing


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')

    root = '/dataset/khtt/dataset/pine2022/ECOM'
    dataset_name = 'small'
    parser.add_argument('--input', help='input', default = osp.join(root,'2.labled', dataset_name))
    parser.add_argument('--voc', help='data processing path')
    parser.add_argument('--output', help='output')
    # parser.add_argument('--IoU', nargs='+', type=float, default = [0.0, 1.0, 0.1], help = 'IoU threhsold to calculate AUC curve')
    parser.add_argument('--IoU', type=float, default = 0.5, help = 'IoU threhsold to calculate AUC curve')
    parser.add_argument('--model_path', help='model_path')
    parser.add_argument('--model', default='unet', help='model used [unet, deeplabv3]')



    # parser.add_argument('--input', help='input')
    # parser.add_argument('--voc', help='data processing path')
    # parser.add_argument('--output', help='output')
    # parser.add_argument('--IoU', nargs='+', type=float, default = [0.0, 1.0, 0.1], help = 'IoU threhsold to calculate AUC curve')
    # parser.add_argument('--model_path', help='model_path')
    # parser.add_argument('--model', default='unet', help='model used [unet, deeplabv3]')

    parser.add_argument('--threshold', default='0.5', help='threshold')
    parser.add_argument('--config',  help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help=('if specified, the evaluation metric results will be dumped'
              'into the directory as json'))
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
             'useful when you want to format the result to a specific format and '
             'submit it to the test server')
    parser.add_argument(
        '--eval',
        default='mIoU',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
             ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
             '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
             'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help="--options is deprecated in favor of --cfg_options' and it will "
             'not be supported in version v0.22.0. Override some settings in the '
             'used config, the key-value pair in xxx=yyy format will be merged '
             'into config file. If the value to be overwritten is a list, it '
             'should be like key="[a,b]" or key=a,b It also allows nested '
             'list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation '
             'marks are necessary and that no white space is allowed.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--compare_method', type=str, default='mask')

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options. '
            '--options will not be supported in version v0.22.0.')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options. '
                      '--options will not be supported in version v0.22.0.')
        args.cfg_options = args.options

    # args.checkpoint = args.model_path

    args.output = '%s_%s'%(args.output, datetime.now().strftime("%Y%m%d_%H%M%S"))

    print('Version: {}'.format(VERSION), '=' * 50)
    pprint.pprint(args, indent=6)
    return args

def read_directory(directory_name, txt_save_path):
    '''
    loop folder and generate the  eval.txt according to the tif file name
    Args:
        directory_name:
        txt_save_path:

    Returns:

    '''
    fw = open(txt_save_path, "w")
    for filename in os.listdir(directory_name):
        #         print(filename)
        if 'jpg' in filename or 'png' in filename or 'tif' in filename:
            # filename = filename[:-4]
            filename = filename.split('.')[0]
            fw.write(filename + '\n')


def maskt2polygon(imbgrs_root, imgrays_root, savepath):
    names = []
    for name in os.listdir(imbgrs_root):
        if name.split('.')[-1].lower() in ['tif', 'jpg', 'png']:
            names.append(name.split('.')[0])

    for name in tqdm(names, desc='Draw polygon:'):
        imbgr_path, imgray_path = osp.join(imbgrs_root, '%s.jpg' % name), osp.join(imgrays_root, '%s.jpg' % name)
        if osp.exists(imbgr_path) and osp.exists(imgray_path):
            imbgr, imgray = cv2.imread(imbgr_path), cv2.imread(imgray_path)

            img_gray = cv2.cvtColor(imgray, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(imbgr, contours, -1, (0, 0, 255), 2)

            cv2.imwrite(osp.join(savepath, "%s.jpg" % name), imbgr)
        else:
            warnings.warn("Image is not exist ! %s, %s"%(imbgr_path, imgray_path))

def mask(args):
    '''
    inference to obtain the predict polygon
    Args:
        args:

    Returns:
    '''


    # mmsegmentation
    if args.model == 'unet':
        configs = '/home/jovyan/model/jbnu/my_model/unet/fcn_unet_s5-d16_128x128_40k_chase_db1.py'
        pred_mask_dir = f'{args.output}/unet_mask/'
        # configs = '/home/zeta1996/job_jsc_ai_2022_serving_mmsegmentation/model/jbnu/my_model/unet/fcn_unet_s5-d16_128x128_40k_chase_db1.py'
    elif args.model == 'deeplabv3':
        configs = '/home/jovyan/model/jbnu/my_model/deeplabv3plus/deeplabv3plus_r50-d8_512x512_20k_voc12aug.py'
        pred_mask_dir = f'{args.output}/deeplabv3_mask/'
    else:
        print('You cannot use this model')

    # read all parameters finish

    # data processing
    root = args.input
    target_path = '%s_input'%args.voc
    if os.path.exists(target_path) == True:
        print('Need not data processing!')
    else:
        data_processing(root, target_path)
    images_path = osp.join(target_path, 'JPEGImages')
    txt_save_path = osp.join(target_path, 'test.txt')
    read_directory(images_path, txt_save_path)


    config_file = configs
    checkpoint_file = args.model_path

    # model init
    model = init_segmentor(config_file, checkpoint_file, device='cuda')

    img_root = images_path

    # predict and generate the mask
    os.makedirs(pred_mask_dir, exist_ok=True)
    for root, _, filenames in os.walk(img_root):
        for filename in tqdm(filenames, desc='Make mask:'):
            name, suffix = filename.split('.')
            if suffix.lower() in ['tif', 'jpg', 'jpeg', 'bmp', 'png']:
                img_path = osp.join(root, filename)
                # print('main loop', img_path)
                result = inference_segmentor(model, img_path)[0]  # > args.threshold
                img = Image.fromarray(np.uint8(result * 55))
                img.save(osp.join(pred_mask_dir, '%s.jpg' % name))
    Logger.info('The inference has been finished, and store the predict results into %s' % pred_mask_dir)


    return pred_mask_dir, target_path



    # maskt2polygon(img_root, pred_mask_dir, args.output)

def text_save(filename, qwe):
    file = open(filename, 'a')
    for i in range(len(qwe)):
        s = str(qwe[i])
        file.write(s)
    file.close()


def main():
    args = parse_args()
    assert args.out or args.eval or args.format_only or args.show \
           or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    # mmsegmentation
    dataname = args.input.split('/')[-1]
    if args.model == 'unet':
        configs = '/home/jovyan/model/jbnu/my_model/unet/fcn_unet_s5-d16_128x128_40k_chase_db1.py'
        maskpath = f'/home/jovyan/logs/unet_{dataname}/unet_mask/'
        # configs = '/home/zeta1996/job_jsc_ai_2022_serving_mmsegmentation/model/jbnu/my_model/unet/fcn_unet_s5-d16_128x128_40k_chase_db1.py'
    elif args.model == 'deeplabv3':
        configs = '/home/jovyan/model/jbnu/my_model/deeplabv3plus/deeplabv3plus_r50-d8_512x512_20k_voc12aug.py'
        maskpath = f'/home/jovyan/logs/deeplabv3_{dataname}/deeplabv3_mask/'
        # configs = '/home/zeta1996/job_jsc_ai_2022_serving_mmsegmentation/model/jbnu/my_model/deeplabv3plus/deeplabv3plus_r50-d8_512x512_20k_voc12aug.py'
    else:
        print('You cannot use this model')

    # read all parameters finish

    # data processing
    root = args.input
    target_path = '%s_input'%args.voc
    if os.path.exists(target_path) == True:
        print('Need not data processing!')
    else:
        data_processing(root, target_path)
    images_path = osp.join(target_path, 'JPEGImages')
    txt_save_path = osp.join(target_path, 'test.txt')
    read_directory(images_path, txt_save_path)


    args.config = configs
    args.work_dir = args.output

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    # print(args.config)
    # if not osp.exists(args.config):
    #     args.config = args.config.replace('/home/jovyan', '/home/khtt/code/insitute_demo/unet_deploy')


    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # write image root path to cfg
    cfg.data.train.data_root = target_path
    cfg.data.val.data_root = target_path
    cfg.data.test.data_root = target_path

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test:
        # hard code index
        cfg.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        ]
        cfg.data.test.pipeline[1].flip = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    if args.gpu_id is not None:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        cfg.gpu_ids = [args.gpu_id]
        distributed = False
        if len(cfg.gpu_ids) > 1:
            warnings.warn(f'The gpu-ids is reset from {cfg.gpu_ids} to '
                          f'{cfg.gpu_ids[0:1]} to avoid potential error in '
                          'non-distribute testing time.')
            cfg.gpu_ids = cfg.gpu_ids[0:1]
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

        json_file = osp.join(args.work_dir,
                             f'eval_single_scale_{timestamp}.json')


    elif rank == 0:
        work_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(args.config))[0])
        mmcv.mkdir_or_exist(osp.abspath(work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(work_dir,
                             f'eval_single_scale_{timestamp}.json')

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    # The default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        shuffle=False)
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
    test_loader_cfg = {
        **loader_cfg,
        'samples_per_gpu': 1,
        'shuffle': False,  # Not shuffle by default
        **cfg.data.get('test_dataloader', {})
    }
    # build the dataloader
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    args.checkpoint = args.model_path
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        print('"CLASSES" not found in meta, use dataset.CLASSES instead')
        model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        print('"PALETTE" not found in meta, use dataset.PALETTE instead')
        model.PALETTE = dataset.PALETTE

    # clean gpu memory when starting a new evaluation.
    torch.cuda.empty_cache()
    eval_kwargs = {} if args.eval_options is None else args.eval_options

    # Deprecated
    efficient_test = eval_kwargs.get('efficient_test', False)
    if efficient_test:
        warnings.warn(
            '``efficient_test=True`` does not have effect in tools/test.py, '
            'the evaluation and format results are CPU memory efficient by '
            'default')

    eval_on_format_results = (
            args.eval is not None and 'cityscapes' in args.eval)
    if eval_on_format_results:
        assert len(args.eval) == 1, 'eval on format results is not ' \
                                    'applicable for metrics other than ' \
                                    'cityscapes'
    if args.format_only or eval_on_format_results:
        if 'imgfile_prefix' in eval_kwargs:
            tmpdir = eval_kwargs['imgfile_prefix']
        else:
            tmpdir = '.format_cityscapes'
            eval_kwargs.setdefault('imgfile_prefix', tmpdir)
        mmcv.mkdir_or_exist(tmpdir)
    else:
        tmpdir = None

    cfg.device = get_device()
    if not distributed:
        warnings.warn(
            'SyncBN is only supported with DDP. To be compatible with DP, '
            'we convert SyncBN to BN. Please use dist_train.sh which can '
            'avoid this error.')
        if not torch.cuda.is_available():
            assert digit_version(mmcv.__version__) >= digit_version('1.4.4'), \
                'Please use MMCV >= 1.4.4 for CPU training!'
        model = revert_sync_batchnorm(model)
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
        results = single_gpu_test(
            model,
            data_loader,
            args.show,
            args.show_dir,
            False,
            args.opacity,
            pre_eval=args.eval is not None and not eval_on_format_results,
            format_only=args.format_only or eval_on_format_results,
            format_args=eval_kwargs)
    else:
        model = build_ddp(
            model,
            cfg.device,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False)
        results = multi_gpu_test(
            model,
            data_loader,
            args.tmpdir,
            args.gpu_collect,
            False,
            pre_eval=args.eval is not None and not eval_on_format_results,
            format_only=args.format_only or eval_on_format_results,
            format_args=eval_kwargs)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            warnings.warn(
                'The behavior of ``args.out`` has been changed since MMSeg '
                'v0.16, the pickled outputs could be seg map as type of '
                'np.array, pre-eval results or file paths for '
                '``dataset.format_results()``.')
            print(f'\nwriting results to {args.out}')
            mmcv.dump(results, args.out)
        if args.eval:
            eval_kwargs.update(metric=args.eval)
            metric = dataset.evaluate(results, **eval_kwargs)
            metric_dict = dict(config=args.config, metric=metric)
            mmcv.dump(metric_dict, json_file, indent=4)
            if tmpdir is not None and eval_on_format_results:
                # remove tmp dir when cityscapes evaluation
                shutil.rmtree(tmpdir)

    print('----------------------------------------------------------------------------------------------------------')

    # inference to generate the mask
    pred_mask_dir, target_path = mask(args)

    # mask convert to json
    polygon_dict = record_coordinate2json(pred_mask_dir, args, target_path, 'eval')
    # obtain the tp, fp, tn, fn to calculate the precision, recall, f1-score

    # import sys
    # sys.exit(0)
    acc, f1 = cal_f1_scoreByIoU(polygon_dict)


    # acc, f1 = f1_score(args.model, args.voc, args.output, args.IoU)

    f = open(json_file, "r", encoding="utf-8")
    a1 = json.load(f)

    txt = osp.join(args.output,'평가기록.txt')

    result = {}
    result['Acc'], result['mAcc'], result['mIoU'], result['IoU'], result['f1-score'], result['oAcc'] = a1['metric']['aAcc'], \
         a1['metric']['mAcc'], a1['metric']['mIoU'], a1['metric']['IoU.diseases'], f1, acc
    print(result)

    with open(txt, 'w', encoding='utf-8') as f:
        f.write('%s'%result)


    try:
        os.remove(json_file)
        shutil.rmtree(pred_mask_dir)
    except:
        pass


if __name__ == '__main__':
    main()