from PIL import Image

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
from model import ft_net, ft_net_dense, ft_net_NAS, PCB, PCB_test
from apex.fp16_utils import *




torch.cuda.set_device(0)
cudnn.benchmark = True

data_transforms = transforms.Compose([
        transforms.Resize((256,128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

######################################################################
# Load model
#---------------------------
def load_network(network):
    network.load_state_dict(torch.load("./model/ft_ResNet50/net_last.pth"))
    return network


######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model, image):
    features = torch.FloatTensor()
    img = data_transforms(image)
    print(img)
    img.unsqueeze_(0)
    n, c, h, w = img.size()
    ff = torch.FloatTensor(1,512).zero_().cuda()

    for i in range(2):
        if(i==1):
            img = fliplr(img)
        input_img = Variable(img.cuda())
        #print(input_img)
        #torch.cat(input_img, )
        model.eval()
        outputs = model(input_img) 
        ff += outputs
    
    # norm feature
    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
    ff = ff.div(fnorm.expand_as(ff))

    features

    features = torch.cat((features,ff.data.cpu()), 0)
    return features


stride = 0.5
model_structure = ft_net(751, stride)

model = load_network(model_structure)


model.classifier.classifier = nn.Sequential()

# Change to test mode
model = model.cuda()

image = Image.open("../testdir/WIN_20200218_11_23_35_Pro (2).jpg")

# Extract feature
with torch.no_grad():
    query_feature = extract_feature(model, image)

print(query_feature)