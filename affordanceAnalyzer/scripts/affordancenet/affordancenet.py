#!/usr/bin/env python3
import sys
import numpy as np
import os, cv2
import argparse
import rospy

file_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_dir)

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from lib.mask_rcnn import MaskRCNNPredictor, MaskAffordancePredictor, MaskRCNNHeads
import lib.mask_rcnn as mask_rcnn
import utils
import argparse
from PIL import Image
import numpy as np
import cv2
import time
from config import IITAFF
import os
from skimage.transform import resize

from affordance_analyzer.srv import *
from std_msgs.msg import MultiArrayDimension, String


class AffordanceAnalyzer(object):
    """docstring for AffordanceAnalyzer."""

    def __init__(self):

        self.CONF_THRESHOLD = 0.7
        self.root_path = '/workspace/affordanceNet_sytn/'
        self.name = "affordancenet_synth"
        self.device = None

        #self.serviceGet = rospy.Service('/affordance/result', getAffordanceSrv, self.getAffordance)
        #self.serviceRun = rospy.Service('/affordance/run', runAffordanceSrv, self.analyzeAffordance)
        #self.serviceStart = rospy.Service('/affordance/start', startAffordanceSrv, self.startAffordance)
        #self.serviceStop = rospy.Service('/affordance/stop', stopAffordanceSrv, self.stopAffordance)
        #self.serviceName = rospy.Service('/affordance/name', getNameSrv, self.getName)

        self.net = None

    def getName(self):
        """
        response = getNameSrvResponse()
        response.name = String(self.name)

        return response
        """
        return self.name


    def run_net(self, x, CONF_THRESHOLD = 0.7):

        #ori_height, ori_width, _ = im.shape

        # Detect all object classes and regress object bounds
        ts = time.time()

        predictions = self.net(x)[0]
        boxes, labels, scores, masks = predictions['boxes'], predictions['labels'], predictions['scores'], predictions['masks']

        te = time.time()
        print("Prediction took: ", te - ts, " seconds")
        try:
            idx = scores > CONF_THRESHOLD
            labels = labels.cpu().detach().numpy()


            boxes = boxes[idx].cpu().detach().numpy()
            labels = labels[idx.cpu().detach().numpy()]
            scores = scores[idx].cpu().detach().numpy()
            masks = masks[idx].cpu().detach().numpy()
            masks = masks * 255

        except:
            pass

        try:
            return boxes, labels, masks, scores
        except:
            return 0

    def analyze_affordance(self, img, CONF_THRESHOLD):

        width, height = img.shape[1], img.shape[0]
        ratio = width / height
        img = cv2.resize(img, (int(450 * ratio), 450), interpolation = cv2.INTER_AREA)
        x = [torchvision.transforms.ToTensor()(img).to(self.device)]
        self.CONF_THRESHOLD = CONF_THRESHOLD

        print("Analyzing affordance with confidence threshold: ", self.CONF_THRESHOLD)
        try:
            bbox, objects, masks, scores = self.run_net(x, CONF_THRESHOLD=self.CONF_THRESHOLD)
            num_objects = masks.shape[0]

            m = np.zeros((num_objects, 11, height, width))
            try:
                for count, mask in enumerate(masks):
                    mask_aff = np.zeros((11, height, width))
                    for count_a, aff_mask in enumerate(mask):
                        mask_aff[count_a] = resize(aff_mask, (height, width))
                    m[count] = mask_aff
                        #m = np.vstack((m,m_aff))
            except Exception as e:
                print(e)
                raise
                return False
            masks = m

            # Arg max the outputs
            t = time.time()
            m = np.zeros(masks.shape)
            for i in range(masks.shape[0]):
                mask_arg = np.argmax(masks[i], axis = 0)
                color_idxs = np.unique(mask_arg)
                for color_idx in color_idxs:
                    if color_idx != 0:
                        idx = i, color_idx, mask_arg == color_idx
                        m[i, color_idx, mask_arg == color_idx] = 1

            masks = m

            for b_c, box in enumerate(bbox):
                box[0] = box[0] * (width / (450 * ratio))
                box[2] = box[2] * (width / (450 * ratio))
                box[1] = box[1] * (height / 450)
                box[3] = box[3] * (height / 450)
                bbox[b_c] = box

            self.bbox = bbox.astype(np.uint32)
            self.objects = objects.astype(np.uint8)
            self.masks = masks.astype(np.uint8)
            self.scores = scores
        except:
            self.bbox = np.zeros((1, 4))
            self.objects = np.zeros((1,1))
            self.masks = np.zeros((1, 11, 244, 244))
            self.scores = np.zeros(1)

        return True

    def getAffordance(self):

        return self.bbox, self.objects, self.masks, self.scores

    def startAffordance(self, use_GPU = True):
        #weights_path = os.path.dirname(os.path.realpath(__file__)) + "/14.pth"
        weights_path = os.path.dirname(os.path.realpath(__file__)) + "/weights.pth"

        device = torch.device('cpu')

        if use_GPU:
            device = torch.device('cuda')
        self.device = device
        print("Device is: ", self.device)

        try:

            model = utils.get_model_instance_segmentation(23, 11)
            #model = utils.get_model_instance_segmentation(18, 8)
            model.load_state_dict(torch.load(weights_path))
            model.to(device)
            model.eval()

            # load network
            self.net = model

            return True

        except:
            return False

    def stopAffordance(self, msg):

        try:
            del self.net
            self.net = None

            return 1

        except:
            return 0


if __name__ == '__main__':

    rospy.init_node('affordance_analyzer')
    affordanceServer = AffordanceAnalyzer()
    print("Affordance analyzer is ready.")

    rospy.spin()
