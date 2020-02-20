# -*- coding: utf-8 -*-

from __future__ import print_function, division

import logging
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
import math
from PIL import Image
from model import ft_net, ft_net_dense, ft_net_NAS, PCB, PCB_test

#fp16
try:
    from apex.fp16_utils import *
except ImportError: # will be 3.x series
    logging.warning('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
######################################################################
# Options
# --------
class feature_extraction:

    def __init__(self, gpu_ids, model_name, ms, batch_size):
        logging.basicConfig(filename='feature_extractor_class.log', level=logging.INFO)

        t_start_init = time.time_ns()
        self.gpu_ids_arg = gpu_ids
        self.name = model_name
        self.ms = ms
        self.which_epoch = 'last'

        # load the training config
        config_path = os.path.join('./model',self.name,'opts.yaml')
        with open(config_path, 'r') as stream:
                config = yaml.load(stream,Loader=yaml.FullLoader)
        self.fp16 = config['fp16'] 
        self.PCB = config['PCB']
        self.use_dense = config['use_dense']
        self.use_NAS = False
        self.stride = config['stride']

        if 'nclasses' in config: # tp compatible with old config files
            self.nclasses = config['nclasses']
        else: 
            self.nclasses = 751 

        str_ids = self.gpu_ids_arg.split(',')

        self.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >=0:
                self.gpu_ids.append(id)

        #print('We use the scale: %s'%self.ms)
        str_ms = self.ms.split(',')
        self.ms = []
        for s in str_ms:
            s_f = float(s)
            self.ms.append(math.sqrt(s_f))

        # set gpu ids
        if len(self.gpu_ids)>0:
            torch.cuda.set_device(self.gpu_ids[0])
            cudnn.benchmark = True
                

        ######################################################################
        # Load Data
        # ---------
        #
        # We will use torchvision and torch.utils.data packages for loading the
        # data.
        #
        self.data_transforms = transforms.Compose([
                transforms.Resize((256,128), interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if self.PCB:
            self.data_transforms = transforms.Compose([
                transforms.Resize((384,192), interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
            ])


        self.use_gpu = torch.cuda.is_available()

        self.load_model()
        #start model optimization for 1 image
        imgArray = []
        img = Image.open("../testdir/WIN_20200218_12_11_00_Pro.jpg")
        for i in range(batch_size):
            imgArray.append(img)
        self.extract_feature(imgArray)
        t_start_init = time.time_ns()
        self.extract_feature(imgArray)
        t_end_init = time.time_ns()
        self.batch_time = (t_end_init - t_start_init)/1000000000
        logging.debug("Batch time is" + str(self.batch_time) + "seconds")
        print(self.batch_time)
        #print("Init took " + str((t_end_init - t_start_init)/1000000000) + " seconds")


    def get_batchtime(self):
        return self.batch_time

    def get_batchsize(self):
        return self.batch_size

    ######################################################################
    # Load model
    #---------------------------
    def load_network(self, network):
        save_path = os.path.join('./model',self.name,'net_%s.pth'%self.which_epoch)
        network.load_state_dict(torch.load(save_path))
        #if self.use_gpu:
            #network.to(torch.device("cuda"))
        return network


    ######################################################################
    # Extract feature
    # ----------------------
    #
    # Extract feature from  a trained model.
    #
    def fliplr(self, img):
        '''flip horizontal'''
        inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
        img_flip = img.index_select(3,inv_idx)
        return img_flip

    def extract_feature(self, image_array):
        features = torch.FloatTensor()
        count = 0
        tensor_list = []

        for image in image_array:
            image_to_add = self.data_transforms(image)
            image_to_add.unsqueeze_(0)
            tensor_list += image_to_add

        img = torch.stack(tensor_list)
        n, c, h, w = img.size()
        count += n
        ff = torch.FloatTensor(n,512).zero_().cuda()
        if self.PCB:
            ff = torch.FloatTensor(n,2048,6).zero_().cuda() # we have six parts

        for i in range(2):

            if(i==1):
                img = self.fliplr(img)
                
            input_img = Variable(img.cuda())
            for scale in self.ms:
                if scale != 1:
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                t_start_eval = time.time_ns()
                outputs = self.model(input_img)
                t_end_eval = time.time_ns()
                logging.debug("Evaluation took " + str((t_end_eval-t_start_eval)/1000000000) + " seconds")
                ff += outputs
        # norm feature
        if self.PCB:
            # feature size (n,2048,6)
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6) 
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features,ff.data.cpu()), 0)
        return features.numpy()

    def extract_feature_numpy(self, numpy_array):
        features = torch.FloatTensor()
        count = 0
        tensor_list = []

        for numpy in numpy_array:
            image = Image.fromarray(numpy)
            image_to_add = self.data_transforms(image)
            image_to_add.unsqueeze_(0)
            tensor_list += image_to_add

        img = torch.stack(tensor_list)
        n, c, h, w = img.size()
        count += n
        ff = torch.FloatTensor(n,512).zero_().cuda()
        if self.PCB:
            ff = torch.FloatTensor(n,2048,6).zero_().cuda() # we have six parts

        for i in range(2):

            if(i==1):
                img = self.fliplr(img)
                
            input_img = Variable(img.cuda())
            for scale in self.ms:
                if scale != 1:
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                t_start_eval = time.time_ns()
                outputs = self.model(input_img)
                t_end_eval = time.time_ns()
                logging.debug("Evaluation took " + str((t_end_eval-t_start_eval)/1000000000) + " seconds")
                ff += outputs
        # norm feature
        if self.PCB:
            # feature size (n,2048,6)
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6) 
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features,ff.data.cpu()), 0)
        return features.numpy()

    ######################################################################
    # Load Collected data Trained model
    def load_model(self):
        if self.use_dense:
            model_structure = ft_net_dense(self.nclasses)
        elif self.use_NAS:
            model_structure = ft_net_NAS(self.nclasses)
        else:
            model_structure = ft_net(self.nclasses, stride = self.stride)

        if self.PCB:
            model_structure = PCB(self.nclasses)

        #if self.fp16:
        #    model_structure = network_to_half(model_structure)

        self.model = self.load_network(model_structure)

        # Remove the final fc layer and classifier layer
        if self.PCB:
            #if self.fp16:
            #    model = PCB_test(model[1])
            #else:
                self.model = PCB_test(self.model)
        else:
            #if self.fp16:
                #model[1].model.fc = nn.Sequential()
                #model[1].classifier = nn.Sequential()
            #else:
                self.model.classifier.classifier = nn.Sequential()

        # Change to test mode
        self.model = self.model.eval()
        if self.use_gpu:
            self.model = self.model.cuda()