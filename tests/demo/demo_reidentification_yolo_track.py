#!/usr/bin/env python

# system imports
import pickle
import sys
import select
import os
import argparse
import numpy as np
from pathlib import Path
import cv2


# personal imports
from camera_utils.cameras.IntelRealsense import IntelRealsense
from ai_utils.detectors.Yolov8InferTrack_old import Yolov8InferTrack
from ai_utils.Reidentificator import Reidentificator
from ai_utils.HandPoseInference import HandPoseInference


parser = argparse.ArgumentParser(
    description='Yolact Inference')
parser.add_argument('--camera_type', default='REALSENSE', type=str,
                    help='RGBD camera')
args = parser.parse_args()


# function used to stop loop functions
def stop_loop(stop_entry: str) -> bool:
    '''
    Used to quit an infinite loop with a char/string entry
    '''
    rlist = select.select([sys.stdin], [], [], 0.001)[0]
    if rlist and sys.stdin.readline().find(stop_entry) != -1:
        return True
    return False


if __name__ == '__main__':

    if args.camera_type=='REALSENSE':
      camera = IntelRealsense(camera_resolution=IntelRealsense.Resolution.HD)
    elif args.camera_type=='ZED':
      camera = Zed(camera_resolution=Zed.Resolution.HD)
    else:
      sys.exit("Wrong camera type!")

    # load diplay ros params
    display_img_results = True
    
    # load yolo network
    yolo_weights= str(Path.home()) + "/Documents/weights/yolov8l-seg.pt"
    reid_weights= str(Path.home()) + "/Documents/weights/osnet_x0_25_msmt17.pt"
    detector = Yolov8InferTrack(display_img=False, return_img=True, model_weights=yolo_weights, reid_weights=reid_weights)


    # Load reidentificator
    mmt_weights = str(Path.home()) + "/Documents/weights/old_pytorch_resnet_ibn_REID_feat256_train_msmt17.pth"
    reident = Reidentificator(class_target="person", display_img=display_img_results, model_weights=mmt_weights)

    # load gesture detection model
    hand_pose = HandPoseInference(display_img=display_img_results)
    hand_weights = str(Path.home()) + "/Documents/weights/right_hand_model.sav"
    hand_classifier = pickle.load(open(hand_weights, 'rb'))


    stop_char = 'q'  # char used to stop the infinite loops

    # Person calibration loop
    # start_time = time.time()
    while True:
        rgb = camera.get_rgb()
        rgb = np.array(rgb)
        det_infer, yolo_img = detector.img_inference(rgb)
        cv2.namedWindow("Yolo", cv2.WINDOW_NORMAL)
        cv2.imshow("Yolo", yolo_img)
        if cv2.waitKey(1) == ord('q'):
                  print("Closed Yolo Image Viewer.")
                  exit(0)
        if reident.calibrate_person(rgb, det_infer) or stop_loop(stop_char):
            break



    while not stop_loop(stop_char):
        color_frame = camera.get_rgb()

        # Person detection
        det_infer, yolo_img = detector.img_inference(color_frame)
        cv2.namedWindow("Yolo", cv2.WINDOW_NORMAL)
        cv2.imshow("Yolo", yolo_img)
        if cv2.waitKey(1) == ord('q'):
                  print("Closed Yolo Image Viewer.")
                  exit(0)


        # Person reidentification
        reidentified_person = reident.reidentify(color_frame, det_infer)

        # if no person re-identified restart detection step
        if reidentified_person is None:
            continue

        reidentified_mask = reidentified_person["masks"]
        reidentified_box = reidentified_person["boxes"]

        hand_img = color_frame.copy()
        hand_img = hand_img[reidentified_box[1]:reidentified_box[3], reidentified_box[0]:reidentified_box[2], :]

        # initialize the prediction class as the last class is the predictor
        gesture_prediction = hand_classifier.n_support_.shape[0] - 1

        hand_results = hand_pose.get_hand_pose(hand_img)
        if hand_results is not None:
            for hand_label in hand_results.keys():
                if hand_label == "Left":
                    continue
                else:
                    gesture_prediction = hand_classifier.predict([hand_results[hand_label]])
                   
        if gesture_prediction == 0:           
            print("OPEN HAND")
        elif gesture_prediction == 1:
            print("CLOSED HAND")
        else:
            print("NOT CLASSIFIED")

