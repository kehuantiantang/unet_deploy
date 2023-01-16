# Copyright 2015 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Python implementation of the GRPC helloworld.Greeter server."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from concurrent import futures
import logging
import argparse

import sys

sys.path.append("/home/jovyan/bizlogic/mywork/job-jsc-v1/grpc")

import grpc
import imgproc_pb2
import imgproc_pb2_grpc

import json
import threading

import numpy as np
from PIL import Image

# from geonet_jsc_unet_infer import JSCUnetInfer

from mmseg.apis import init_segmentor, inference_segmentor

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')

    # PWD parameters
    parser.add_argument('--model', default='unet', help='model used [unet, deeplabv3]')
    parser.add_argument('--img_path', default='/home/zeta1996/job_jsc_ai_2022_serving_mmsegmentation/datasets/train_sample_datasets_input/JPEGImages/base_zl_20_tx_892781_ty_410208__from_3s_222_243_02.jpg', help='path to input data')
    parser.add_argument('--model_path', default='/home/zeta1996/job_jsc_ai_2022_serving_mmsegmentation/model/weights/unet/iter_20000.pth', help='model used [unet, deeplabv3]')
    parser.add_argument(
        '--gpu_num',
        type=int,
        default=0,
        help='id of gpu to use '
             '(only applicable to non-distributed testing)')
    args = parser.parse_args()

    return args

sema = threading.Semaphore(1)

args = parse_args()

# mmsegmentation
if args.model == 'unet':
    configs = '/home/jovyan/model/jbnu/my_model/unet/fcn_unet_s5-d16_128x128_40k_chase_db1.py'
    # configs = '/home/zeta1996/job_jsc_ai_2022_serving_mmsegmentation/model/jbnu/my_model/unet/fcn_unet_s5-d16_128x128_40k_chase_db1.py'
elif args.model == 'deeplabv3':
    configs = '/home/jovyan/model/jbnu/my_model/deeplabv3plus/deeplabv3plus_r50-d8_512x512_20k_voc12aug.py'
    # configs = '/home/zeta1996/job_jsc_ai_2022_serving_mmsegmentation/model/jbnu/my_model/deeplabv3plus/deeplabv3plus_r50-d8_512x512_20k_voc12aug.py'
else:
    print('You cannot use this model')


model = init_segmentor(configs, args.model_path, device='cuda:'+str(args.gpu_num))



class Service(imgproc_pb2_grpc.ImgProcServicer):

    def do(self, request, context):
        print("Start")
        reqobj = json.loads(request.reqobj)

        rcv_img = None
        rcv_img2 = None

        # print( request.img2 )

        if request.img != b'':
            rcv_img = np.frombuffer(request.img, np.uint8)
            rcv_img = rcv_img.reshape(reqobj["img"]["width"], reqobj["img"]["height"], reqobj["img"]["channels"])
        if request.img2 != b'':
            rcv_img2 = np.frombuffer(request.img2, np.uint8)
            rcv_img2 = rcv_img2.reshape(reqobj["img2"]["width"], reqobj["img2"]["height"], reqobj["img2"]["channels"])

        sema.acquire()
        rst_img = inference_segmentor(model, args.img_path)
        sema.release()

        resobj = {}
        if rst_img is not None:
            resobj["img"] = {"height": rst_img.shape[1], "width": rst_img.shape[0], "channels": 1}

        return imgproc_pb2.ImgProcReply(result='ok', resobj=json.dumps(resobj), img=rst_img.tobytes(), img2=None)


def serve():
    MAX_MESSAGE_LENGTH = 3 * 1024 * 1024 * 3

    # server = grpc.server( futures.ThreadPoolExecutor(max_workers=10), options )
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
                         options=[("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH)],
                         )

    imgproc_pb2_grpc.add_ImgProcServicer_to_server(Service(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


# python ./model/serve/serve.py  --model_path= --gpu_num="0,1,2"
if __name__ == '__main__':
    logging.basicConfig()
    serve()