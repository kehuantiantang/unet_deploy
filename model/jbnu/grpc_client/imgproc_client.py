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
"""The Python implementation of the GRPC helloworld.Greeter client."""

from __future__ import print_function

import logging

import grpc
import imgproc_pb2
import imgproc_pb2_grpc

import cv2
import numpy as np
import json


def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    
    maxMsgLength = 3 * 1024 * 1024 * 3 
    options=[('grpc.max_message_length', maxMsgLength),
                          ('grpc.max_send_message_length', maxMsgLength),
                          ('grpc.max_receive_message_length', maxMsgLength)]
        
    with grpc.insecure_channel('localhost:50051', options) as channel:
        
        stub = imgproc_pb2_grpc.ImgProcStub(channel )
        
        img = cv2.imread('test.jpg', cv2.IMREAD_COLOR)
        img2 = cv2.imread('test.jpg', cv2.IMREAD_COLOR)
        
        reqobj = {}
        reqobj["img"]  = { "height" : img.shape[1] ,  "width": img.shape[0],  "channels" : 3  }
        reqobj["img2"] = { "height" : img2.shape[1] , "width": img2.shape[0], "channels" : 3  }
                
        response = stub.do( imgproc_pb2.ImgProcRequest( cmd = 'unet', reqobj = json.dumps(reqobj), img = img.tobytes(),  img2 = img2.tobytes() ) )
                
        resobj = json.loads( response.resobj )
        print( response.resobj )        
        
        if "img" in resobj:  
            rcv_img  = np.frombuffer(response.img , np.uint8)
            rcv_img  = rcv_img.reshape( resobj["img"]["width"], resobj["img"]["height"], resobj["img"]["channels"] )
            print( rcv_img.shape )              
            cv2.imwrite('rcv_test.jpg', rcv_img )
        
        if "img2" in resobj:  
            rcv_img2 = np.frombuffer(response.img2, np.uint8)
            rcv_img2 = rcv_img2.reshape( resobj["img2"]["width"], resobj["img2"]["height"], resobj["img2"]["channels"] ) 
            print( rcv_img2.shape )      
            cv2.imwrite('rcv_test2.jpg', rcv_img2)
        
    print("Greeter client received: " + response.result )


if __name__ == '__main__':
    logging.basicConfig()
    run()
