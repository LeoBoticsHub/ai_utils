import math
import os
import pdb
import sys

from pathlib import Path

from camera_utils.cameras.IntelRealsense import IntelRealsense
from ai_utils.detectors.Yolov8InferTrack import Yolov8InferTrack
from ai_utils.detectors.DetectorOutput import DetectorOutput

import numpy as np
import cv2
import time
import argparse


if __name__ == '__main__':
    yolo_weights = str(Path.home()) + "/Downloads/yolov8l-seg.pt"
    reid_weights = str(Path.home()) + "/Downloads/osnet_x0_25_msmt17.pt"
    
    yolo = Yolov8InferTrack(display_img=True, return_img=True, model_weights=yolo_weights, reid_weights=reid_weights)

    camera = IntelRealsense(camera_resolution=IntelRealsense.Resolution.HD)

    prove = 0 
    somma = 0
    
    while True:
        start_time = time.time()
        prove+=1
        img = camera.get_rgb()

        infer: DetectorOutput
        infer = yolo.img_inference(img)
        print(1/(time.time()-start_time))
        somma += (time.time()-start_time)

        if infer:
            cv2.namedWindow("YOLO TRACKING RETURNED", cv2.WINDOW_NORMAL)
            cv2.imshow('YOLO TRACKING RETURNED', infer.image)
            if cv2.waitKey(1) == ord('q'):  
                break

    mean_time = somma/prove
    freq = 1/mean_time

    print("\nmean time: %.4f \nfrequency: %.4f \n" % (mean_time, freq))




