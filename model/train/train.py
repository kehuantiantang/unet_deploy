# Copyright (c) OpenMMLab. All rights reserved.
# coding=utf-8
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
sys.path.append('./')
import argparse
import copy
import os.path as osp
import time
import warnings
from model.preprocessing.logger import Logger, VERSION
from model.preprocessing.logger import self_print as print
import mmcv
import torch
import torch.distributed as dist
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import Config, DictAction, get_git_hash
from mmseg import __version__
from mmseg.apis import init_random_seed, set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import (collect_env, get_device, get_root_logger,
                         setup_multi_processes)

from model.preprocessing.preprocess import data_processing


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    
    parser.add_argument('--input', default='', help='input data')
    parser.add_argument('--model', default='unet', help='model used [unet, deeplabv3]')
    parser.add_argument('--output', default='', help='path to output model')
    parser.add_argument('--voc', help='data processing path')

    parser.add_argument('--config', default='', help='train config file path')
    parser.add_argument('--work-dir', default='', help='the dir to save logs and models')


    parser.add_argument(
        '--load-from', default='/home/jovyan/model/weights/MODEL_NAME/iter_20000.pth', help='the checkpoint file to '
                                                                                           'load '
                                                                                   'weights from')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-num',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--diff_seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
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
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically.')
    args = parser.parse_args()

    args.load_from = args.load_from.replace('MODEL_NAME', args.model)

    if not os.path.exists(args.load_from):
        print('load_from is not exist, set to empty', args.load_from)
        args.load_from = ''


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


    return args

def read_directory(directory_name, txt_save_path):
    fw = open(txt_save_path, "w", encoding = 'utf-8')
    for filename in os.listdir(directory_name):
        #         print(filename)
        filename = filename[:-4]
        fw.write(filename + '\n')



def main():
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

    args = parse_args()

    # data preprocessing
    root = args.input
    # args.input_pro = args.input + '_input/'
    target_path = '%s_input'%args.voc
    # target_path = root + '_input/'
    # os.makedirs(target_path, exist_ok=True)
    if os.path.exists(target_path) == True:
        print('Need not data processing!')
    else:
        data_processing(root, target_path)
    images_path = osp.join(target_path, 'JPEGImages')
    txt_save_path = osp.join(target_path, 'train.txt')
    read_directory(images_path, txt_save_path)
    
    # mmsegmentation
    if args.model == 'unet':
        configs = '/home/jovyan/model/jbnu/my_model/unet/fcn_unet_s5-d16_128x128_40k_chase_db1.py'
        # configs = '/home/zeta1996/job_jsc_ai_2022_serving_mmsegmentation/model/jbnu/my_model/unet/fcn_unet_s5-d16_128x128_40k_chase_db1.py'
    elif args.model == 'deeplabv3':
        configs = '/home/jovyan/model/jbnu/my_model/deeplabv3plus/deeplabv3plus_r50-d8_512x512_20k_voc12aug.py'
    elif args.model == 'segformer-b5':
        configs = '/home/jovyan/model/jbnu/my_model/segformer/segformer_mit-b5.py'
    else:
        print('You cannot use this model')

    args.config = configs


    #     read config file
    cfg = Config.fromfile(args.config)
    #  the information has store in variable in cfg

    # write image root path to cfg
    cfg.data.train.data_root = target_path
    cfg.data.val.data_root = target_path
    cfg.data.test.data_root = target_path


    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
        
    args.work_dir = args.output
    args.gpu_id = args.gpu_num

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.work_dir = '%s-%s'%(cfg.work_dir, timestamp)
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support '
                      'single GPU mode in non-distributed training. '
                      'Use `gpus=1` now.')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed training. Use the first GPU '
                      'in `gpu_ids` now.')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    cfg.auto_resume = args.auto_resume

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # gpu_ids is used to calculate iter when resuming checkpoint
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)



    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))



    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    Logger.init(log_file=log_file, logfile_level='debug', stdout_level=cfg.log_level)
    Logger.debug(configs)




    # set multi-process settings
    setup_multi_processes(cfg)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    Logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    Logger.info(f'Distributed training: {distributed}')
    Logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    cfg.device = get_device()
    seed = init_random_seed(args.seed, device=cfg.device)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    Logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    # SyncBN is not support for DP
    if not distributed:
        warnings.warn(
            'SyncBN is only supported with DDP. To be compatible with DP, '
            'we convert SyncBN to BN. Please use dist_train.sh which can '
            'avoid this error.')
        model = revert_sync_batchnorm(model)

    Logger.info(model)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmseg version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    # passing checkpoint meta for saving best checkpoint
    meta.update(cfg.checkpoint_config.meta)
    train_segmentor(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
