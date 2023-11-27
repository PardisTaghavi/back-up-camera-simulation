#!/usr/bin/env python

import os
import sys
current_dir = os.getcwd()
sys.path.append(current_dir+'/yolov7')

from configparser import Interpolation
import sys
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from PIL import Image as im
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import  load_classifier, TracedModel
from utils.datasets import letterbox

weights=current_dir+'/yolov7/yolov7.pt' #adjust the path if needed
source=0  
img_size=640
conf_thres=0.7
iou_thres=0.5
device=torch.device('cuda:0') #torch.cuda.current_device() #cuda device, i.e. 0 or 0,1,2,3 or cpu'
view_img=True
save_conf=True
nosave=True
classes=None #'filter by class: --class 0, or --class 0 2 3'
agnostic_nms=False
augment=True
no_trace=True #'don`t trace model'
trace=False

class detections():

    def __init__(self):
        self.weights=weights
        self.device=device
        self.img_size=img_size
        self.iou_thres=iou_thres
        self.augment=augment
        self.view_img=view_img
        self.classes=classes
        self.agnostic_nms=agnostic_nms
        self.w=0
        self.h=0
        self.conf_thres=conf_thres
        self.classify=False
        self.half=self.device !='cpu' #half precision
    
        # Load model 
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        print("model is here", self.model)
        self.stride = int(self.model.stride.max())  # model stride
        print("stride", self.stride)
        self.img_size = check_img_size(self.img_size, s=self.stride)  # check img_size
        print("image size is", self.img_size)

        if trace:
            self.model = TracedModel(self.model, device, self.img_size)
            print("model is traced")

        if self.half:
            self.model.half()  # to FP16
            print("model is halfed")
       
        cudnn.benchmark = True  # set True to speed up constant image size inference

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        print("names and colors are set")

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.img_size, self.img_size).to(self.device).type_as(next(self.model.parameters())))  # run once
            print("interference is running")
        

    def detection(self, frame):
        
        img0=frame
        img=letterbox(img0, self.img_size, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  #BGR to RGB
        img = np.ascontiguousarray(img)
        img=self.preProccess(img)
        
        with torch.no_grad():

            pred=self.model(img, augment=self.augment)[0]
            pred=non_max_suppression(pred, conf_thres=self.conf_thres, iou_thres=self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)

            for i,det in enumerate(pred):
                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        if view_img:
                            label = f'{self.names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, img0, label=label, color=self.colors[int(cls)], line_thickness=3)

        return img0

    def preProccess(self, img):
        imgs=torch.from_numpy(img).to(self.device)
        img=imgs.half() if self.half else img.float()  # uint8 to fp16/32
        img=img/255.0
        if img.ndimension()==3:
            img=img.unsqueeze(0)
        return img
        

