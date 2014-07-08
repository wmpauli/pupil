'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2014  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import sys, os,platform
import cv2
import numpy as np
from file_methods import Persistent_Dict
from gl_utils import draw_gl_points

from methods import normalize,denormalize
from glfw import *
import atb
from ctypes import c_int,c_bool,c_float

from plugin import Plugin
#logging
import logging
logger = logging.getLogger(__name__)

# from reference_surface import Reference_Surface
# from math import sqrt

class Natural_Feature_Detector(Plugin):
    """docstring
    """
    def __init__(self,g_pool=None,size=2,color=(0.,1.0,0.5,.5),atb_pos=(320,200)):
        super(Natural_Feature_Detector, self).__init__()
        self.g_pool = g_pool
        self.order = .2

        self.atb_pos = atb_pos
        self.size = c_int(int(size))
        self.color = (c_float*4)(*color)

        # for comparison on speed vs. robustness see:
        # http://computer-vision-talks.com/articles/2011-01-04-comparison-of-the-opencv-feature-detection-algorithms/

        # There are 3 different implementations to access feature detectors/extractors
        # detectors/extractor packages:
        # - cv2.ORB, cv2.SURF, cv2.SIFT, cv2.BRISK
        # detector
        # - cv2.FeatureDetector_create("DETECTORNAME")
        # - detectors:
        #       - FAST, STAR, SIFT, SURF, ORB, BRISK, MSER, GFTT, HARRIS, SIMPLEBLOB
        # extractor
        # - cv2.DescriptorExtractor_create("EXTRACTORNAME")
        #   extractors:
        #       - SIFT, SURF, BRIEF, ORB, FREAK
        # direct access to a specific package
        # - cv2.FastFeatureDetector()

        # self.detector = cv2.ORB( nfeatures = 500 )
        self.detector = cv2.FeatureDetector_create("FAST")
        self.extractor = cv2.DescriptorExtractor_create("FREAK")
        # self.detector = cv2.FastFeatureDetector(threshold=200)

        # features detected 
        self.features = []


    def init_gui(self,pos=None):
        pos = self.atb_pos
        import atb
        atb_label = "natural feature detector"
        self._bar = atb.Bar(name =self.__class__.__name__, label=atb_label,
            help="natural feature detector parameters", color=(50, 150, 50), alpha=100,
            text='light', position=pos,refresh=.3, size=(300, 150))

        self._bar.add_var('color',self.color)
        self._bar.add_button('remove',self.unset_alive)    

    def unset_alive(self):
        self.alive = False

    def detect_features(self, frame):
        '''detect_features(self, frame) -> keypoints, descrs'''
        keypoints, descrs = self.detector.detectAndCompute(frame, None)
        if descrs is None:  # detectAndCompute returns descs=None if not keypoints found
            descrs = []
        return keypoints, descrs


    def update(self,frame,recent_pupil_positions,events):
        # kp, descrs = self.detect_features(frame.img)
        kp = self.detector.detect(frame.img, None)
        self.kp, des = self.extractor.compute(frame.img, kp)        

        # kp = self.detector.detect(frame.img, None)
        # pts = [denormalize(k.pt,frame.img.shape[:-1][::-1],flip_y=True) for k in kp]
        # for k in kp:
           # transparent_circle(frame.img, k.pt, radius=radius, color=color, thickness=thickness)

    def gl_display(self):
        """
        Display marker and surface info inside world screen
        """
        size = self.size.value
        color = self.color[::]
        draw_gl_points([k.pt for k in self.kp],size=size,color=color)



    def cleanup(self):
        """ called when the plugin gets terminated.
        This happends either voluntary or forced.
        if you have an atb bar or glfw window destroy it here.
        """
        self._bar.destroy()










