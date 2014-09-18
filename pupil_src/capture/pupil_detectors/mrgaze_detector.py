'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2014  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import cv2
from time import sleep
import numpy as np
from methods import *
import atb
from ctypes import c_int,c_bool,c_float
import logging
logger = logging.getLogger(__name__)

from mrgaze import utils, config
from mrgaze.pupilometry import PupilometryEngine
import ConfigParser
import os

class MrGaze_Detector(object):
    """ a pupil detector based on Mr. Gaze """
    dummy_param = c_int(0)
    no_result = {}
    no_result['norm_pupil'] = None

    def __init__(self, gpool):
        super(MrGaze_Detector, self).__init__()
        # Create a new parser
        self.cfg = ConfigParser.ConfigParser()
        self.cfg = config.InitConfig(self.cfg)

        mrclean_root = utils._package_root()
        LBP_path = os.path.join(mrclean_root, 'Cascade/cascade.xml')
        logger.debug('  Loading LBP cascade')
        self.cascade = cv2.CascadeClassifier(LBP_path)

    def detect(self,frame,user_roi,visualize=False):
        img = frame.img
        # hint: create a view into the img with the bounds of user set region of interest
        pupil_img = img[user_roi.lY:user_roi.uY,user_roi.lX:user_roi.uX]
        if visualize:
            pass
            # draw into image whatever you like and it will be displayed
            # otherwise you shall not modify img data inplace!


        # call Mike's amazing PupilometryEngine
        gray_img = cv2.cvtColor(pupil_img,cv2.COLOR_BGR2GRAY)

        p_r = Roi(gray_img.shape)
        p_r.set((0,0,None,None))
        w = img.shape[0]/2
        
        e, roi_rect, blink, glint = PupilometryEngine(gray_img, self.cascade, self.cfg)
        pupil_ellipse = {}
        pupil_ellipse['confidence'] = .9
        pupil_ellipse['ellipse'] = e
        pupil_ellipse['roi_center'] = e[0]
        pupil_ellipse['major'] = max(e[1])
        pupil_ellipse['minor'] = min(e[1])
        pupil_ellipse['apparent_pupil_size'] = max(e[1])
        pupil_ellipse['axes'] = e[1]
        pupil_ellipse['angle'] = e[2]
        e_img_center = user_roi.add_vector(p_r.add_vector(e[0]))
        norm_center = normalize(e_img_center,(frame.img.shape[1], frame.img.shape[0]),flip_y=True)
        pupil_ellipse['norm_pupil'] = norm_center
        pupil_ellipse['center'] = e_img_center
        pupil_ellipse['timestamp'] = frame.timestamp

        if np.isnan(e[0][0]) == False:
            return pupil_ellipse # all this will be sent to the world process, you can add whateever you need to this.

        else:
            self.no_result['timestamp'] = frame.timestamp
            return self.no_result


    def create_atb_bar(self,pos):
        self.bar = atb.Bar(name = "Pupil_Detector", label="Pupil Detector Controls",
            help="pupil detection params", color=(50, 50, 50), alpha=100,
            text='light', position=pos,refresh=.3, size=(200, 200))
        self.bar.add_var("DUMMY_PARAM",self.dummy_param, step=1.,readonly=False)



