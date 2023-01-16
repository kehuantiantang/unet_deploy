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

from concurrent import futures
import logging

import grpc
import imgproc_pb2
import imgproc_pb2_grpc


class Service(imgproc_pb2_grpc.ImgProcServicer):

    def do(self, request, context):
        # 这里写自己的业务代码
        print("grpc server 进来了")
        # 这里是返回
        return imgproc_pb2.ImgProcReply( result = 'ok', resobj = request.reqobj, img = request.img ,  img2 = request.img2  )


def serve():
    
    MAX_MESSAGE_LENGTH = 3 * 1024 * 1024 * 3 
            
    #server = grpc.server( futures.ThreadPoolExecutor(max_workers=10), options )
    server = grpc.server( futures.ThreadPoolExecutor(max_workers=10),
        options=[("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH)],
    )
    
    imgproc_pb2_grpc.add_ImgProcServicer_to_server(Service(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    serve()