import cv2
import numpy as np
import torch
from torch.nn import functional as F
import sys
from mmt import models
from mmt.utils.serialization import load_checkpoint, copy_state_dict

from pathlib import Path
home_path = str(Path.home())


class FastBaseTransform(torch.nn.Module):
    """
    Transform that does all operations on the GPU for super speed.
    This doesn't support a lot of config settings and should only be used for production.
    Maintain this as necessary.
    """
    MEANS = (103.94, 116.78, 123.68)
    STD = (57.38, 57.12, 58.40)

    def __init__(self):
        super().__init__()
        self.mean = torch.Tensor(self.MEANS).float().cuda()[None, :, None, None]
        self.std = torch.Tensor(self.STD).float().cuda()[None, :, None, None]

    def forward(self, img):
        self.mean = self.mean.to(img.device)
        self.std = self.std.to(img.device)
        # img assumed to be a pytorch BGR image with channel order [n, h, w, c]
        img_size = (256, 128)
        img = img.permute(0, 3, 1, 2).contiguous()
        img = F.interpolate(img, img_size, mode='bilinear', align_corners=False)
        img = (img - self.mean) / self.std
        img = img[:, (2, 1, 0), :, :].contiguous()
        # Return value is in channel order [n, c, h, w] and RGB
        return img


weights_path = home_path + '/weights/resnet_ibn_REID.tar'


class Reidentificator:
    '''
    Reidentify an object on an image using the inference output of another AI algorithm and calibration
    '''

    def __init__(self, class_target, display_target=False, model_weights=weights_path):
        '''
        initialize the Re-identificator object
        :param class_target: the class of the object you want to track
        :param display_target: boolean value to return an image which create a bounding box around the
        re-identified object
        '''

        self.class_target = class_target
        self.display_target = display_target

        self.transform = torch.nn.DataParallel(FastBaseTransform()).cuda()
        print('Loading REID model...', end='')
        self.model_REID = models.create('resnet_ibn50a', pretrained=False, num_features=0, dropout=0, num_classes=0)
        self.model_REID.cuda()
        self.model_REID = torch.nn.DataParallel(self.model_REID)
        try:
            checkpoint = load_checkpoint(model_weights)
        except ValueError:
            print('\n\033[91mWeights not found in ' + model_weights + ". You must download them "
                                                                      "in that directory.\033[0m")
            exit(1)
        copy_state_dict(checkpoint['state_dict'], self.model_REID)
        self.model_REID.eval()
        print('Done.')

        self.seq_n = 0
        self.meas_init = 100
        self.calibrated = False

        self.person_avg_feat = 0  # template feature

    def calibrate_person(self, rgb, inference_output):
        '''
        Function used to calibrate the reidentificator with the object image. This function should be called iteratively
        until it returns True (i.e., when the object is calibrated)
        :param rgb: the image in which there is the object
        :param inference_output: a dictionary containing the inferences obtained by an instance segmentation algorithm
        (e.g., Yolact++)
        :return: A boolean which confirm if the object has been correctly calibrated or not
        '''
        try:
            boxes = inference_output[self.class_target]['boxes']
        except KeyError:
            return self.calibrated

        if len(boxes) > 1:
            print('WARNING: MORE THAN ONE PERSON DETECTED DURING CALIBRATION!')
            self.person_avg_feat = 0
            self.seq_n = 0
        else:
            percentage = int(self.seq_n / self.meas_init * 100)
            if percentage % 10 == 0:
                print("CALIBRATING ", percentage, "%")
            img_person = self.transform(
                torch.from_numpy(rgb[boxes[0][1]:boxes[0][3], boxes[0][0]:boxes[0][2], :]).unsqueeze(0).cuda().float())
            self.seq_n += 1
            self.person_avg_feat += self.model_REID(img_person.cuda()).data.cpu()

            # after meas_init measures we terminate the initialization
            if self.seq_n > self.meas_init:
                print('\nCALIBRATION FINISHED')
                self.calibrated = True
                self.person_avg_feat = self.person_avg_feat / np.linalg.norm(self.person_avg_feat)
                self.seq_n = 0

        return self.calibrated

    def reidentify(self, rgb, inference_output):
        '''
        Used to reidentify the calibrated object on the image (if present)
        :param rgb: the image in which there should be the object to reidentify
        :param inference_output: a dictionary containing the inferences obtained by an instance segmentation algorithm
        (e.g., Yolact++)
        :return: the image with a bounding box (depending on self.display_target) and the mask of the targeted object
        reidentified
        '''
        rgb = rgb.copy()
        if not self.calibrated:
            sys.exit("Error: Reidentificator not calibrated!")

        try:
            boxes = inference_output[self.class_target]['boxes']
            masks = inference_output[self.class_target]['masks']
        except KeyError:
            print("No target detected")
            return rgb, None

        img_persons = []
        ### COPY THE FEATURES TEMPLATE ACCORDINGLY TO NUMBER OF DETECTED PERSONS FOR FAST DISTANCE COMPUTATION
        person_avg_feat_temp = np.tile(self.person_avg_feat, (len(boxes), 1))
        ### CUT THE BOUNDING BOXES OF THE DETECTED PERSONS OVER THE IMAGE
        for id in range(len(boxes)):
            person_bb = self.transform(
                torch.from_numpy(rgb[boxes[id][1]:boxes[id][3], boxes[id][0]:boxes[id][2], :]).unsqueeze(
                    0).cuda().float())
            img_persons.append(person_bb[0])
        img_persons = [img_person.cuda().float() for img_person in img_persons]
        img_persons = torch.stack(img_persons, 0)
        ### PASS THE IMAGES INSIDE THE EXTERNAL NETWORK
        feat_pers = self.model_REID(img_persons).data.cpu()
        ### COMPUTE FEATURES DISTANCES
        dist = np.linalg.norm(feat_pers - person_avg_feat_temp, axis=1)
        if np.min(dist)>1:
          return rgb, None
        target_idx = np.argmin(dist)
        # print(np.sort(dist))
        # Todo: if dist è maggiore di una threshold allora non tornare nulla
        if self.display_target:
            x1, y1, x2, y2 = boxes[target_idx]
            cv2.rectangle(rgb, (x1, y1), (x2, y2), (255, 255, 255), 5)

            text_str = 'TARGET'

            font_face = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.6
            font_thickness = 1

            text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

            text_pt = (x1, y1 - 3)
            text_color = [255, 255, 255]

            cv2.rectangle(rgb, (x1, y1), (x1 + text_w, y1 - text_h - 4), (0, 0, 0), -1)
            cv2.putText(rgb, text_str, text_pt, font_face, font_scale, text_color, font_thickness,
                        cv2.LINE_AA)

        return rgb, masks[target_idx]