#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 02:34:14 2016

@author: chernov
"""

import shutil
from os import environ, makedirs, path, system

import caffe
from caffe import layers as L
from caffe.proto import caffe_pb2
from google.protobuf import text_format


NET_NAME = "rubli"
NET_PATH = environ["BEORGDATAGEN"] + "/%s_net"%(NET_NAME)
NET_PROTO_TRAIN = NET_PATH + "/%s_train.prototxt"%(NET_NAME)
NET_PROTO_TEST = NET_PATH + "/%s_test.prototxt"%(NET_NAME)
NET_PROTO_DEPLOY = NET_PATH + "/%s_net.prototxt"%(NET_NAME)
SOLVER_PROTO = NET_PATH + "/%s_net_solver.prototxt"%(NET_NAME)
TRAIN_HDF5_PATH = environ["BEORGDATAGEN"] + \
                  "/CData_full/train_%s.h5.list.txt"%(NET_NAME)
TEST_HDF5_PATH = environ["BEORGDATAGEN"] + \
                 "/CData_full/test_%s.h5.list.txt"%(NET_NAME)

                 
def define_solver_params():
    params = caffe_pb2.SolverParameter()
    params.train_net = NET_PROTO_TRAIN
    params.test_net.append(NET_PROTO_TEST)
    params.test_iter.append(100)
    params.test_interval = 500
    params.base_lr = 0.001
    params.momentum = 0.9
    params.weight_decay = 0.001
    params.lr_policy = "inv"
    params.gamma = 0.01
    params.power =  0.75
    params.display = 500
    params.max_iter = 30000
    params.snapshot = 5000
    params.snapshot_prefix = NET_NAME
    return params
    
    
def add_base_layers(n):
    
    n.conv1 = L.Convolution(bottom="data", 
                           num_output=20,
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
                           num_output=50,
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
    n.ip2 = L.InnerProduct(n.relu1, num_output=8,
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
    
    
def define_net_test(batch_size=64):
    n = caffe.NetSpec()
    n.data, n.label = L.HDF5Data(batch_size=batch_size,
                                 source=TEST_HDF5_PATH, ntop=2)
    n = add_base_layers(n)
    n.loss = L.EuclideanLoss(n.ip2, n.label)
    return n.to_proto()

    
def define_net_deploy():
    n = caffe.NetSpec() 
    n = add_base_layers(n)
    
    proto = 'input: "data"\n' \
            'input_shape {\n' \
            '  dim: 1 # batchsize\n' \
            '  dim: 3 # number of colour channels - rgb\n' \
            '  dim: 30 # width\n' \
            '  dim: 15 # height\n' \
            '}\n'
            
    proto += text_format.MessageToString(n.to_proto())
    return proto

    
def create_solver_params_proto():
    params = define_solver_params()
    solver_proto = text_format.MessageToString(params)
    with open(SOLVER_PROTO, 'w') as f:
        f.write(solver_proto)


if not path.exists(environ["BEORGDATAGEN"] + "/CData_full"):
    print("%s not found, start croping"%(environ["BEORGDATAGEN"] + "/CData_full"))
    system("python3 ../marking_tools/crop_pricers.py")
    print("croping done")
else:
    print("%s found, skip croping. To force restart delete this folder"
          %(environ["BEORGDATAGEN"] + "/CData_full"))

if not path.exists(TRAIN_HDF5_PATH):
    print("%s not found, generaing hdf5"%(TRAIN_HDF5_PATH))
    system("python3 ../marking_tools/generate_hdf5.py %s"%(NET_NAME))
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
        train_proto = text_format.MessageToString(define_net_test())
        f.write(train_proto)
        
    with open(NET_PROTO_DEPLOY, 'w') as f:
        train_proto = define_net_deploy()
        f.write(train_proto)

