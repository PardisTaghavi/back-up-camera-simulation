import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import os
import sys

from detection import detections
from models.experimental import attempt_load
from utils.torch_utils import select_device



detections_= detections()


def simulate_rear_wheel_trajectory(frame, radius, wheelbase, length):
    
    
    # Wheelbase (distance between front and rear axles)
    L = 2.5  
    B = 1.5

    # constant theta for right and left wheels
    theta = np.pi/6

    # alpha is the angle between the center of the circle and the center of the rear axle 
    # "i" shows the inner wheel and "o" shows the outer wheel
    center_alpha = np.arctan2(length, radius)
    alpha_i = np.arctan2(wheelbase, radius - length/2) + theta
    alpha_o = np.arctan2(wheelbase, radius + length/2) - theta
 
    #assume the car is in the middle of the frame
    # zeros are initial coordinates of the rear axle
    x0, y0  = frame.shape[1]/2 , frame.shape[0]
    # add dynamic offset to x 
    offset = 255
    xi0, yi0 = x0 + offset, y0
    xo0, yo0 = x0 - offset, y0 

    x_i = xi0 - radius * np.sin(alpha_i)
    y_i = yi0 - radius * np.cos(alpha_i)

    x_o = xo0 - radius * np.sin(alpha_o)
    y_o = yo0 - radius * np.cos(alpha_o)

    x_c = x0 - radius * np.sin(center_alpha)
    y_c = y0 - radius * np.cos(center_alpha)

    trajectory_c = np.array([[x0, y0], [x_c, y_c]], np.int32)
    trajectory_i = np.array([[xi0, yi0], [x_i, y_i]], np.int32)
    trajectory_o = np.array([[xo0, yo0], [x_o, y_o]], np.int32)

    return trajectory_c, trajectory_i, trajectory_o


def overlay_trajectory(video_path):

    # radius of the circle, wheelbase and length of the car assumed to be known
    radius = 200
    wheelbase = 2.5
    length = 1.5

    cap = cv2.VideoCapture(video_path)
    if (cap.isOpened()== False): 
      print("Error opening video stream or file")

    # Read until video is completed
    while(cap.isOpened()):
      
      # Capture frame-by-frame
      ret, frame = cap.read()
      if ret == True:

        # Display the resulting frame
        trajectory_c, trajectory_i, trajectory_o = simulate_rear_wheel_trajectory(frame, radius, wheelbase, length)
        cv2.polylines(frame, [trajectory_c], False, (0, 255, 0), 2)
        cv2.polylines(frame, [trajectory_i], False, (0, 0, 255), 2)
        cv2.polylines(frame, [trajectory_o], False, (0, 0, 255), 2)

        # object detection
        input_img = frame        
        detections_.detection(frame=input_img) #predictions include coordinates of the bounding boxes, the class of the object and the confidence score
        
        cv2.imshow('Frame',frame)
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
          break
      else: 
        break

    cap.release()
    cv2.destroyAllWindows()

    return
    



video_path= '/home/avalocal/Desktop/code/IMG_7467.mp4'
steering_angle = np.pi/6
wheelbase = 2.5

overlay_trajectory(video_path)
