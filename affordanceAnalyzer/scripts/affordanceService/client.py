#!/usr/bin/env python3

import time
import rospy
import cv2
import numpy as np

from affordancenet.affordancenet import AffordanceAnalyzer

class AffordanceClient(object):
    """docstring for AffordanceClient."""

    def __init__(self, connected = True):

        self.no_objects = 0
        self.masks = None
        self.bbox = None
        self.objects = None
        self.scores = None
        self.GPU = False

        self.affNet = AffordanceAnalyzer()

        self.name = "affordancenet_synth"
        if connected:
            self.name = self.getName()

        if self.name == "affordancenet":
            self.noObjClass = 11
            self.noLabelClass = 10
            self.OBJ_CLASSES = ('__background__', 'bowl', 'tvm', 'pan', 'hammer', 'knife',
                                    'cup', 'drill', 'racket', 'spatula', 'bottle')
            self.OBJ_NAME_TO_ID = {'__background__': 0, 'bowl': 1, 'tvm': 2, 'pan': 3,
                                'hammer': 4, 'knife': 5, 'cup': 6, 'drill': 7,
                                'racket': 8, 'spatula': 9, 'bottle': 10}
            self.labelsToNames = {'__background__': 0, 'contain': 1, 'cut': 2, 'display': 3, 'engine': 4, 'grasp': 5, 'hit': 6, 'pound': 7, 'support': 8, 'w-grasp': 9}
            self.graspLabels = [5, 9]
            self.functional_labels = [1, 2, 3, 4, 6, 7, 8]

        elif self.name == "affordancenet_context":
            self.noObjClass = 2
            self.noLabelClass = 8
            self.OBJ_CLASSES = ('__background__', 'objectness')
            self.OBJ_NAME_TO_ID = {'__background__': 0, 'objectness': 1}
            self.labelsToNames = {'__background__': 0, 'grasp': 1, 'cut': 2, 'scoop': 3, 'contain': 4, 'pound': 5, 'support': 6, 'w-grasp': 7}
            self.graspLabels = [1, 7]
            self.functional_labels = [2, 3, 4, 5, 6]

        elif self.name == "affordancenet_synth":
            self.noObjClass = 23
            self.noLabelClass = 11
            self.OBJ_CLASSES = ('__background__', 'knife', 'saw', 'scissors', 'shears', 'scoop',
                                        'spoon', 'trowel', 'bowl', 'cup', 'ladle',
                                        'mug', 'pot', 'shovel', 'turner', 'hammer',
                                        'mallet', 'tenderizer', 'bottle', 'drill', 'monitor', 'pan', 'racket')
            self.OBJ_NAME_TO_ID = {'__background__': 0, 'knife': 1, 'saw': 2, 'scissors': 3, 'shears': 4, 'scoop': 5,
                                'spoon': 6, 'trowel': 7, 'bowl': 8, 'cup': 9, 'ladle': 10,
                                'mug': 11, 'pot': 12, 'shovel': 13, 'turner': 14, 'hammer': 15,
                                'mallet': 16, 'tenderizer': 17, 'bottle': 18, 'drill': 19, 'tvm': 20, 'pan': 21, 'racket': 22}
            self.NamesToLabels = {'__background__': 0, 'grasp': 1, 'cut': 2, 'scoop': 3, 'contain': 4, 'pound': 5, 'support': 6, 'wrap-grasp': 7, 'display': 8, 'engine': 9, 'hit': 10}
            self.labelsToNames = {0: '__background__', 1: 'grasp', 2: 'cut', 3: 'scoop', 4: 'contain', 5: 'pound', 6: 'support', 7: 'wrap-grasp', 8: 'display', 9: 'engine', 10: 'hit'}
            self.graspLabels = [1]
            self.functional_labels = [2, 3, 4, 5, 6, 7, 8, 9]

    def getName(self):
        return self.affNet.getName()

    def getFunctionalLabels(self):
        return self.functional_labels


    def start(self, GPU=False):
        self.GPU = GPU

        return self.affNet.startAffordance(use_GPU = GPU)


    def stop(self):
        return self.affNet.stopAffordance()

    def run(self, img, CONF_THRESHOLD = 0.7):
        return self.affNet.analyze_affordance(img, CONF_THRESHOLD)

    def getAffordanceResult(self):

        self.bbox, self.objects, self.masks, self.scores = self.affNet.getAffordance()
        self.no_objects = self.masks.shape[0]

        return self.masks, self.objects, self.scores, self.bbox
