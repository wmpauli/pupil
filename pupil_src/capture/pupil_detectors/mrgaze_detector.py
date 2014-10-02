'''
(*)~--------------------------------------------------------------------------

 Video pupilometry 
 - detects pupils

 AUTHOR : Mike Tyszka, Wolfgang Pauli
 PLACE : Caltech
 DATES : 2014-09-17 WMP from scratch

 This file is part of mrgaze.

 mrgaze is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 mrgaze is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with mrgaze. If not, see <http://www.gnu.org/licenses/>.

 Copyright 2014 California Institute of Technology.

--------------------------------------------------------------------------~(*)
'''

import cv2
from time import sleep, time
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

    no_result = {} # this is returned when no pupil was detected
    no_result['norm_pupil'] = None
    no_result['timestamp'] = time() 

    def __init__(self, gpool): # what's up with gpood. We don't seem to be using it
        ''' Class Initiator) '''
        super(MrGaze_Detector, self).__init__()
        
        # Create a new parser
        self.cfg = ConfigParser.ConfigParser()
        self.cfg = config.InitConfig(self.cfg)
        
        # Init Cascade Classfier
        mrclean_root = utils._package_root()
        LBP_path = os.path.join(mrclean_root, 'Cascade/cascade.xml')
        logger.debug('  Loading LBP cascade')
        self.cascade = cv2.CascadeClassifier(LBP_path)

    def detect(self,frame,user_roi, visualize=False):
        ''' detect a pupil in this frame, taking roi into account '''
        img = frame.img
        # hint: create a view into the img with the bounds of user set region of interest
        pupil_img = img[user_roi.lY:user_roi.uY,user_roi.lX:user_roi.uX]

        # convert to gray scale
        gray_img = cv2.cvtColor(pupil_img,cv2.COLOR_BGR2GRAY)

        # get ROI properties? Not sure we need to call this as often (check canny_detector)
        p_r = Roi(gray_img.shape)
        p_r.set((0,0,None,None))
        w = img.shape[0]/2
        
        # call Mike's amazing PupilometryEngine
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

        # return pupil if we found one, otherwise return no_result
        if np.isnan(e[0][0]) == False:
            return pupil_ellipse # all this will be sent to the world process, you can add whateever you need to this.
        else:
            self.no_result['timestamp'] = frame.timestamp
            return self.no_result


    def set_ldb_minneighbors(self,minneighbors):
        ''' set min neighbors of classifier ''' 
        self.cfg.set('LBP','minneighbors',str(minneighbors))

    def get_ldb_minneighbors(self):
        ''' get setting of min neighbors in classifier '''
        return c_int(self.cfg.getint('LBP','minneighbors'))

    def create_atb_bar(self,pos):
        ''' create advanced tweak bar with setting for Mr. Gaze '''
        self.bar = atb.Bar(name = "Mr_Gaze_Detector", label="Mr. Gaze Controls",
            help="Mr. Gaze Params", color=(50, 50, 50), alpha=100,
            text='light', position=pos, refresh=.3, size=(200, 100))

        self.bar.add_var("min neighbors", vtype=c_int, setter=self.set_ldb_minneighbors, getter=self.get_ldb_minneighbors)



