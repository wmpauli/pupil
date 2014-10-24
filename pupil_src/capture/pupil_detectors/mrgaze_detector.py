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
from ctypes import c_int,c_bool,c_float,c_char_p
import logging
logger = logging.getLogger(__name__)

from mrgaze import utils, config
from mrgaze.pupilometry import PupilometryEngine
from mrgaze.media import Downsample
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
        self.gpool = gpool

        self.cfg = config.LoadConfig(os.path.join(gpool.user_dir))
#        # Create a new parser
#        self.cfg = ConfigParser.ConfigParser()
#        self.cfg = config.InitConfig(self.cfg)
        
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
        
        downsampling = self.cfg.getfloat('VIDEO', 'downsampling')
        if downsampling > 1:
            gray_img = Downsample(gray_img, downsampling)

        # call Mike's amazing PupilometryEngine
        e, roi_rect, blink, glint, rgb_frame = PupilometryEngine(gray_img, self.cascade, self.cfg)
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


    def set_ldb_enabled(self,enabled):
        ''' set min neighbors of classifier ''' 
        self.cfg.set('LBP','enabled', str(enabled))

    def get_ldb_enabled(self):
        ''' get setting of min neighbors in classifier '''
        return c_bool(self.cfg.getboolean('LBP','enabled'))

    def set_ldb_minneighbors(self,minneighbors):
        ''' set min neighbors of classifier ''' 
        self.cfg.set('LBP','minneighbors',str(minneighbors))

    def get_ldb_minneighbors(self):
        ''' get setting of min neighbors in classifier '''
        return c_int(self.cfg.getint('LBP','minneighbors'))

    def set_glint_percmax(self,glint_percmax):
        ''' set min neighbors of classifier ''' 
        self.cfg.set('PUPILSEG','glint_percmax',str(glint_percmax))

    def get_glint_percmax(self):
        ''' get setting of min neighbors in classifier '''
        return c_int(self.cfg.getint('PUPILSEG','glint_percmax'))

    def set_pupil_threshold(self,pupil_threshold):
        ''' set min neighbors of classifier ''' 
        self.cfg.set('PUPILSEG','pupil_threshold',str(pupil_threshold))

    def get_pupil_threshold(self):
        ''' get setting of min neighbors in classifier '''
        return c_int(self.cfg.getint('PUPILSEG','pupil_threshold'))

    def set_pupil_percmax(self,pupil_percmax):
        ''' set max neighbors of classifier ''' 
        self.cfg.set('PUPILSEG','pupil_percmax',str(pupil_percmax))

    def get_pupil_percmax(self):
        ''' get setting of max neighbors in classifier '''
        return c_int(self.cfg.getint('PUPILSEG','pupil_percmax'))

    def set_downsampling(self,downsampling):
        ''' set max neighbors of classifier ''' 
        self.cfg.set('VIDEO','downsampling',str(downsampling))

    def get_downsampling(self):
        ''' get setting of max neighbors in classifier '''
        return c_int(self.cfg.getint('VIDEO','downsampling'))

    def set_ransac_maxiter(self,maxiterations):
        ''' set max neighbors of classifier ''' 
        self.cfg.set('RANSAC','maxiterations',str(maxiterations))

    def get_ransac_maxiter(self):
        ''' get setting of max neighbors in classifier '''
        return c_int(self.cfg.getint('RANSAC','maxiterations'))

    def set_ransac_maxrefine(self,maxrefinements):
        ''' set max neighbors of classifier ''' 
        self.cfg.set('RANSAC','maxrefinements',str(maxrefinements))

    def get_ransac_maxrefine(self):
        ''' get setting of max neighbors in classifier '''
        return c_int(self.cfg.getint('RANSAC','maxrefinements'))

    def set_ransac_maxinlierperc(self,maxinlierperc):
        ''' set max neighbors of classifier ''' 
        self.cfg.set('RANSAC','maxinlierperc',str(maxinlierperc))

    def get_ransac_maxinlierperc(self):
        ''' get setting of max neighbors in classifier '''
        return c_int(self.cfg.getint('RANSAC','maxinlierperc'))

    def set_k_dil(self,k_dil):
        ''' set max neighbors of classifier ''' 
        self.cfg.set('PUPILSEG','k_dil',str(k_dil))

    def get_k_dil(self):
        ''' get setting of max neighbors in classifier '''
        return c_int(self.cfg.getint('PUPILSEG','k_dil'))

    def set_k_inpaint(self,k_inpaint):
        ''' set max neighbors of classifier ''' 
        self.cfg.set('PUPILSEG','k_inpaint',str(k_inpaint))

    def get_k_inpaint(self):
        ''' get setting of max neighbors in classifier '''
        return c_int(self.cfg.getint('PUPILSEG','k_inpaint'))

    def set_gauss_sd(self,gauss_sd):
        ''' set max neighbors of classifier ''' 
        self.cfg.set('PUPILSEG','gauss_sd',str(gauss_sd))

    def get_gauss_sd(self):
        ''' get setting of max neighbors in classifier '''
        return c_int(self.cfg.getint('PUPILSEG','gauss_sd'))

    def set_graphics(self,graphics):
        ''' set max neighbors of classifier ''' 
        self.cfg.set('OUTPUT', 'graphics', str(graphics))

    def get_graphics(self):
        ''' get setting of max neighbors in classifier '''
        return c_bool(self.cfg.getboolean('OUTPUT','graphics'))

    def set_histogram_equalization(self,histogram_equalization):
        ''' set max neighbors of classifier ''' 
        self.cfg.set('PUPILSEG', 'histogram_equalization', str(histogram_equalization))

    def get_histogram_equalization(self):
        ''' get setting of max neighbors in classifier '''
        return c_bool(self.cfg.getboolean('PUPILSEG','histogram_equalization'))

    def set_method(self, method):
        ''' set segmentation method '''
        method_dict = {0:"otsu", 1:"manual", 2: "kmeans"} 
        self.cfg.set('PUPILSEG', 'method', method_dict[method])

    def get_method(self):
        ''' get setting of max neighbors in classifier '''
        method = self.cfg.get('PUPILSEG','method')
        method_dict = {"otsu": 0, "manual": 1, "kmeans": 2} 
        return c_int(method_dict[method])

    def create_atb_bar(self,pos):
        ''' create advanced tweak bar with setting for Mr. Gaze '''
        self.bar = atb.Bar(name = "Mr_Gaze_Detector", label="Mr. Gaze Controls",
            help="Mr. Gaze Params", color=(50, 50, 50), alpha=100,
            text='light', position=pos, refresh=.3, size=(200, 250))
#        self.bar.fps = c_float(10)
#        
#        self.bar.add_var("fps", self.bar.fps, min=1)
        self.bar.add_var("LPB", vtype=c_bool, setter=self.set_ldb_enabled, getter=self.get_ldb_enabled)
        self.bar.add_var("min neighbors", vtype=c_int, setter=self.set_ldb_minneighbors, getter=self.get_ldb_minneighbors)
        self.bar.pupilseg_method_enum = atb.enum("Method",{"otsu":0,
                                                  "manual":1,
                                                  "kmeans":2})

        self.bar.add_var("Method", vtype=self.bar.pupilseg_method_enum, setter=self.set_method, getter=self.get_method, help="select pupil seg method")

        self.bar.add_var("glint_percmax", vtype=c_int, setter=self.set_glint_percmax, getter=self.get_glint_percmax, min=0, max=100)
        self.bar.add_var("pupil_threshold", vtype=c_int, setter=self.set_pupil_threshold, getter=self.get_pupil_threshold, min=0, max=100)
        self.bar.add_var("p_percmax", vtype=c_int, setter=self.set_pupil_percmax, getter=self.get_pupil_percmax, min=0, max=100)
        self.bar.add_var("downsampling", vtype=c_int, setter=self.set_downsampling, getter=self.get_downsampling, min=1, max=100)
        self.bar.add_var("ransac_maxiter", vtype=c_int, setter=self.set_ransac_maxiter, getter=self.get_ransac_maxiter, min=0, max=100)
        self.bar.add_var("ransac_maxrefine", vtype=c_int, setter=self.set_ransac_maxrefine, getter=self.get_ransac_maxrefine, min=0, max=100)
        self.bar.add_var("ransac_maxinlierperc", vtype=c_int, setter=self.set_ransac_maxinlierperc, getter=self.get_ransac_maxinlierperc, min=0, max=100)
        self.bar.add_var("k_inpaint", vtype=c_int, setter=self.set_k_inpaint, getter=self.get_k_inpaint, min=1, max=100)
        self.bar.add_var("k_dil", vtype=c_int, setter=self.set_k_dil, getter=self.get_k_dil, min=1, max=100)
        self.bar.add_var("gauss_sd", vtype=c_int, setter=self.set_gauss_sd, getter=self.get_gauss_sd, min=0, max=100)
        self.bar.add_var("histogram_equalization", vtype=c_bool, setter=self.set_histogram_equalization, getter=self.get_histogram_equalization)
        self.bar.add_var("graphics", vtype=c_bool, setter=self.set_graphics, getter=self.get_graphics)


    def cleanup(self):
        config.SaveConfig(self.cfg, self.gpool.user_dir)
        return 
