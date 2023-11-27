## Backup Camera Simulation:
dynamic model: Ackermann Steering Geometry
object detection model: pretrained yolov7

### Objective:
This project aims to simulate a backup camera. We used dynamic rear wheel trajectory using Ackermann steering geometry. Object detection is implemented using the pre-trained YOLOv7 model. Here's a concise summary of our approach:

### Simulation of Backup Camera View and Rear Wheel Trajectory:

Ackermann Steering Geometry is used to calculate steering angles(δ) based on wheelbase(L) and turning radius(R). Due to the absence of specific sensor data, we assume the bottom middle of the image as the car's reference point. In real scenarios, data of sensors such as GPU or IMU could be used and be transformed to image coordinates using intrinsic and extrinsic camera parameters.

### Object Detection Using YOLOv7:

YOLOv7, pre-trained on COCO dataset, is used for object detection without fine-tuning for this exercise. Results show good detection and real-time performance so we did not fine-tune the model fo this assignment.


### Implementation:


This code use yolov7:
```git clone https://github.com/WongKinYiu/yolov7.git```
```git clone https://github.com/PardisTaghavi/back-up-camera-simulation.git```
video of the backup camera should be downloaded from the videos folder and yolov7 model should be downloaded from the following link: 
https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
run the code through
```python code.py ```

### Conclusion:
This project successfully integrates Ackermann steering geometry for realistic trajectory simulation and employs YOLOv7 for accurate object detection. Assumptions include the reference point and pre-trained YOLOv7 model usage. The code is well-organized, documented, and comes with clear execution instructions.

### Result:
![ezgif com-gif-maker](https://github.com/PardisTaghavi/back-up-camera-simulation/blob/main/results/result.gif)
