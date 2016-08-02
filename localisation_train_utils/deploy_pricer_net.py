#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 02:34:14 2016

@author: chernov
"""

import shutil
from os import environ, makedirs, path, system
import sys

import caffe
from caffe import layers as L
from caffe.proto import caffe_pb2
from google.protobuf import text_format

main_dir = path.abspath(path.dirname(path.dirname(__file__)))
if not main_dir in sys.path:
    sys.path.append(main_dir)

from pricer_net.convertDataToHdf5 import generate_HDF5

RESIZE = (30, 60)
NET_NAME = "pricer"
NET_PATH = environ["BEORGDATAGEN"] + "/%s_net"%(NET_NAME)
NET_PROTO_TRAIN = NET_PATH + "/%s_train.prototxt"%(NET_NAME)
NET_PROTO_TEST = NET_PATH + "/%s_test.prototxt"%(NET_NAME)
NET_PROTO_DEPLOY = NET_PATH + "/%s_net.prototxt"%(NET_NAME)
SOLVER_PROTO = NET_PATH + "/%s_net_solver.prototxt"%(NET_NAME)
QUADRANTS_PATH =  path.join(environ["BEORGDATAGEN"], "croped_quadrants")
TRAIN_HDF5_PATH = path.join(QUADRANTS_PATH, "train.index.txt")
TEST_HDF5_PATH = path.join(QUADRANTS_PATH, "test.index.txt")
                 
def define_solver_params():
    params = caffe_pb2.SolverParameter()
    params.train_net = NET_PROTO_TRAIN
    params.test_net.append(NET_PROTO_TEST)
    params.test_iter.append(100)
    params.test_interval = 1000
    params.base_lr = 0.0001
    params.momentum = 0.9
    params.weight_decay = 0.0005
    params.lr_policy = "inv"
    params.gamma = 0.0001
    params.power =  0.75
    params.display = 500
    params.max_iter = 100000
    params.snapshot = 5000
    params.snapshot_prefix = NET_NAME
    return params
    
def add_base_layers(n):
    return add_base_layers_minst(n)

def add_base_layers_minst(n):
    n.conv1 = L.Convolution(n.data, 
                           num_output=40,
                           kernel_size=5,
                           stride=1,
                           param=[dict(lr_mult=1), dict(lr_mult=2)],
                           weight_filler=dict(type='xavier'),
                           bias_filler=dict(type='constant'))
    n.pool1 = L.Pooling(n.conv1, 
                        kernel_size=2,
                        stride=2,
                        pool=caffe_pb2.PoolingParameter.MAX)
    n.conv2 = L.Convolution(n.pool1, 
                           num_output=100,
                           kernel_size=5,
                           stride=1,
                           param=[dict(lr_mult=1), dict(lr_mult=2)],
                           weight_filler=dict(type='xavier'),
                           bias_filler=dict(type='constant'))
    n.pool2 = L.Pooling(n.conv2, 
                        kernel_size=2,
                        stride=2,
                        pool=caffe_pb2.PoolingParameter.MAX)
    n.ip1 = L.InnerProduct(n.pool2, 
                           param=[dict(lr_mult=1), dict(lr_mult=2)],
                           num_output=500,
                           weight_filler=dict(type='xavier'),
                           bias_filler=dict(type='constant'))
    n.relu1 = L.ReLU(n.ip1)
    n.ip2 = L.InnerProduct(n.relu1, num_output=2,
                           weight_filler=dict(type='xavier'),
                           bias_filler=dict(type='constant'))
    return n
    
def add_base_layers_lenet(n):
    
    n.conv1 = L.Convolution(n.data, 
                           num_output=128,
                           kernel_w=5,
                           kernel_h=10,
                           stride_w=1,
                           stride_h=2,
                           param=[dict(lr_mult=1), dict(lr_mult=2)],
                           weight_filler=dict(type='xavier'),
                           bias_filler=dict(type='constant'))
    n.relu1 = L.ReLU(n.conv1)
    n.pool1 = L.Pooling(n.relu1, 
                        kernel_size=3,
                        stride=1,
                        pool=caffe_pb2.PoolingParameter.MAX)
    
    n.conv2 = L.Convolution(n.pool1, 
                           num_output=256,
                           pad=2,
                           kernel_size=5,
                           param=[dict(lr_mult=1), dict(lr_mult=2)],
                           weight_filler=dict(type='xavier'),
                           bias_filler=dict(type='constant'))
    n.relu2 = L.ReLU(n.conv2)
    n.pool2 = L.Pooling(n.relu2, 
                        kernel_size=3,
                        stride=1,
                        pool=caffe_pb2.PoolingParameter.MAX)
    n.conv3 = L.Convolution(n.pool2, 
                           num_output=512,
                           pad=1,
                           kernel_size=3,
                           param=[dict(lr_mult=1), dict(lr_mult=2)],
                           weight_filler=dict(type='xavier'),
                           bias_filler=dict(type='constant'))
    n.relu3 = L.ReLU(n.conv3)
    n.conv4 = L.Convolution(n.relu3, 
                           num_output=512,
                           pad=1,
                           kernel_size=3,
                           param=[dict(lr_mult=1), dict(lr_mult=2)],
                           weight_filler=dict(type='xavier'),
                           bias_filler=dict(type='constant'))
    n.relu4 = L.ReLU(n.conv4)
    n.pool4 = L.Pooling(n.relu4, 
                        kernel_size=3,
                        stride=1,
                        pool=caffe_pb2.PoolingParameter.MAX)
    n.conv5 = L.Convolution(n.pool4, 
                           num_output=384,
                           pad=1,
                           kernel_size=3,
                           param=[dict(lr_mult=1), dict(lr_mult=2)],
                           weight_filler=dict(type='xavier'),
                           bias_filler=dict(type='constant'))
    n.relu5 = L.ReLU(n.conv5)
    n.pool5 = L.Pooling(n.relu5, 
                        kernel_size=3,
                        stride=1,
                        pool=caffe_pb2.PoolingParameter.MAX)
    n.ip1 = L.InnerProduct(n.pool5, 
                           param=[dict(lr_mult=1), dict(lr_mult=2)],
                           num_output=1024,
                           weight_filler=dict(type='xavier'),
                           bias_filler=dict(type='constant'))
    n.relu1_ip_1 = L.ReLU(n.ip1)
    n.ip2 = L.InnerProduct(n.relu1_ip_1, num_output=2,
                           weight_filler=dict(type='xavier'),
                           bias_filler=dict(type='constant'))
    return n
    
    
def define_net_train(batch_size=64):
    n = caffe.NetSpec()
    n.data, n.label = L.HDF5Data(batch_size=batch_size,
                                 source=TRAIN_HDF5_PATH, ntop=2)
    n = add_base_layers(n)
    n.loss = L.EuclideanLoss(n.ip2, n.label)
    return n.to_proto()
    
    
def define_net_test(batch_size=92):
    n = caffe.NetSpec()
    n.data, n.label = L.HDF5Data(batch_size=batch_size,
                                 source=TEST_HDF5_PATH, ntop=2)
    n = add_base_layers(n)
    n.loss = L.EuclideanLoss(n.ip2, n.label)
    return n.to_proto()

    
def define_net_deploy():
    n = caffe.NetSpec()
    shape = caffe_pb2.BlobShape()
    shape.dim.append(1)
    shape.dim.append(1)
    shape.dim.append(RESIZE[0])
    shape.dim.append(RESIZE[1])
    n.data = L.Input(input_param=(dict(shape=shape)))
    n = add_base_layers(n)            
    return n.to_proto()

    
def create_solver_params_proto():
    params = define_solver_params()
    solver_proto = text_format.MessageToString(params)
    with open(SOLVER_PROTO, 'w') as f:
        f.write(solver_proto)
    
if not path.exists(QUADRANTS_PATH):
    print("%s not found, start croping"%(QUADRANTS_PATH))
    command = "python3 %s %s %s --increase=%s --increase_art=%s"%(
              path.join(main_dir, "marking_tools/gen_pricer_loc_dataset.py"),
              path.join(environ["BEORGDATA"], "localization_ML/"
                                              "output_points.txt"),
              QUADRANTS_PATH, 10, 2)
    system(command)
    print("croping done")
else:
    print("%s found, skip croping. To force restart delete this folder"
          %(QUADRANTS_PATH))
    

if not path.exists(TRAIN_HDF5_PATH):
    print("%s not found, generaing hdf5"%(TRAIN_HDF5_PATH))
    generate_HDF5(path.join(QUADRANTS_PATH, "images"),
                  path.join(QUADRANTS_PATH, "corners.txt"),
                  RESIZE, 1, 10, True, out_dir=QUADRANTS_PATH)
    print("generaing done")
else:
    print("%s found, hdf5. To force recreate delete hdf5 bases"
          %(TRAIN_HDF5_PATH))
    

if(path.exists(NET_PATH)):
    print("%s already exists, to force deploy remove it"%(NET_PATH))
else:
    makedirs(NET_PATH)

    create_solver_params_proto()
    shutil.copyfile("%s_net/get_rect.py"%(NET_NAME), NET_PATH + "/get_rect.py")
    shutil.copyfile("%s_net/train.py"%(NET_NAME), NET_PATH + "/train.py")
    
    with open(NET_PROTO_TRAIN, 'w') as f:
        train_proto = text_format.MessageToString(define_net_train())
        f.write(train_proto)
    
    with open(NET_PROTO_TEST, 'w') as f:
        test_proto = text_format.MessageToString(define_net_test())
        f.write(test_proto)
        
    with open(NET_PROTO_DEPLOY, 'w') as f:
        deploy_proto = text_format.MessageToString(define_net_deploy())
        f.write(deploy_proto)

