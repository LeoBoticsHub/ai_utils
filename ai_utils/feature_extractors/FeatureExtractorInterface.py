'''----------------------------------------------------------------------------------------------------------------------------------
# Copyright (C) 2022
#
# author: Federico Rollo
# mail: rollo.f96@gmail.com
#
# Institute: Leonardo Labs (Leonardo S.p.a - Istituto Italiano di tecnologia)
#
# This file is part of ai_utils. <https://github.com/IASRobolab/ai_utils>
#
# ai_utils is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ai_utils is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License. If not, see http://www.gnu.org/licenses/
---------------------------------------------------------------------------------------------------------------------------------'''
import cv2

class FeatureExtractorInterface:


    def __init__(self, target_classes: list, save_imgs = False) -> None:
        self.set_target_classes(target_classes)
        self.cropped_images = {}
        self.save_imgs = save_imgs

        if save_imgs:
            self.original_images = {}
            self.target_num = 1
            self.path = "/tmp"
            self.distr_num = 1


    '''
    Get features using the orignal image and its detector inference output (a instance segmentation network)
    '''
    def get_features(self, image, detector_inference):
        raise NotImplementedError
    

    '''
    Get features from already cropped images
    '''
    def get_pure_feature(self, images, batch = 8):
        raise NotImplementedError
    
    '''
    Save current network weights. Useful after a training.
    '''
    def save_weights(self, path):
        raise NotImplementedError
    
    
    def set_target_classes(self, target_classes):
        self.target_classes = target_classes


    def get_cropped_imgs(self) -> dict:
        if self.save_imgs:
            for img in list(self.original_images.values()):
                    img_name = "distractor_training_img_" + str(self.distr_num) + ".jpg"
                    self.distr_num += 1
                    cv2.imwrite(self.path + img_name, img)

        return self.cropped_images
    

    def get_cropped_img_by_id(self, idx: int):
        if idx in self.cropped_images.keys():
            if self.save_imgs:
                img_name = "target_training_img_" + str(self.target_num) + ".jpg"
                self.target_num += 1
                cv2.imwrite(self.path + img_name, self.original_images[idx])
            return self.cropped_images[idx]
        return None
    

    def set_network_weights(self, weights):
        raise NotImplementedError

    
    def get_network_weights(self):
        raise NotImplementedError


