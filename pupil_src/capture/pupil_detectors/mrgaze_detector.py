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
from ctypes import c_int,c_bool,c_float,c_char_p,c_long
import logging
logger = logging.getLogger(__name__)

from mrgaze import utils, config
from mrgaze.pupilometry import PupilometryEngine
from mrgaze.media import Preproc
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
        # gray_img = cv2.cvtColor(pupil_img,cv2.COLOR_BGR2GRAY)

        
        #downsampling = self.cfg.getfloat('VIDEO', 'downsampling')
        #if downsampling > 1:
        #    gray_img = Downsample(gray_img, downsampling)

        # call Mike's amazing PupilometryEngine
        pupil_img, art_power = Preproc(pupil_img, self.cfg)
        e, roi_rect, blink, glint, rgb_frame = PupilometryEngine(pupil_img, self.cascade, self.cfg)

        # get ROI properties? Not sure we need to call this as often (check canny_detector)
        p_r = Roi(pupil_img.shape)
        p_r.set((0,0,None,None))
        w = img.shape[0]/2

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
        if np.isnan(e[0][0]) == False and not blink:
            return pupil_ellipse # all this will be sent to the world process, you can add whateever you need to this.
        else:
            self.no_result['timestamp'] = frame.timestamp
            return self.no_result



    # VIDEO section --------------

    def set_downsampling(self,downsampling):
        ''' set amount of downsampling of the input image ''' 
        self.cfg.set('VIDEO','downsampling',str(downsampling))

    def get_downsampling(self):
        ''' get amount of downsampling of the input image ''' 
        return c_int(int(self.cfg.getint('VIDEO','downsampling')))



    # PUPILDETECT section ---------------

    def set_pupildetect_enabled(self, enable):
        ''' enable/disable pupil detection '''
        self.cfg.set('PUPILDETECT', 'enabled', str(enable))

    def get_pupildetect_enabled(self):
        ''' check whether pupil detection is enabled/disabled'''
        return c_bool(self.cfg.getboolean('PUPILDETECT','enabled'))


    def set_specificity(self, specificity):
        ''' set specificity of pupil detection '''
        self.cfg.set('PUPILDETECT', 'specificity', str(specificity))

    def get_specificity(self):
        ''' get specificity of pupil detection '''
        return c_int(self.cfg.getint('PUPILDETECT','specificity'))


    def set_scalefactor(self,scalefactor):
        ''' set scalefactor in pupil detection '''
        self.cfg.set('PUPILDETECT','scalefactor',str(scalefactor))

    def get_scalefactor(self):
        ''' get scalefactor in pupil detection '''
        return c_float(self.cfg.getfloat('PUPILDETECT','scalefactor'))


    
    # PUPILSEG section ---------------

    def set_pupilseg_method(self, method):
        ''' set pupil segmentation method '''
        method_dict = {0:"otsu", 1:"manual"}
        self.cfg.set('PUPILSEG', 'method', method_dict[method])

    def get_pupilseg_method(self):
        ''' get pupil segmentation method '''
        method = self.cfg.get('PUPILSEG','method')
        method_dict = {"otsu": 0, "manual": 1}
        return c_int(method_dict[method])

    
    def set_pupildiameterperc(self, pupildiameterperc):
        ''' set pupildiameterperc in pupil segmentation '''
        self.cfg.set('PUPILSEG','pupildiameterperc',str(pupildiameterperc))

    def get_pupildiameterperc(self):
        ''' get pupildiameterperc in pupil segmentation '''
        return c_float(self.cfg.getfloat('PUPILSEG','pupildiameterperc'))

    
    def set_glintdiameterperc(self, glintdiameterperc):
        ''' set glintdiameterperc in pupil segmentation '''
        self.cfg.set('PUPILSEG','glintdiameterperc',str(glintdiameterperc))

    def get_glintdiameterperc(self):
        ''' get glintdiameterperc in pupil segmentation '''
        return c_float(self.cfg.getfloat('PUPILSEG','glintdiameterperc'))

    
    def set_pupilthresholdperc(self, pupilthresholdperc):
        ''' set pupilthresholdperc in pupil segmentation '''
        self.cfg.set('PUPILSEG','pupilthresholdperc',str(pupilthresholdperc))

    def get_pupilthresholdperc(self):
        ''' get pupilthresholdperc in pupil segmentation '''
        return c_float(self.cfg.getfloat('PUPILSEG','pupilthresholdperc'))

    
    def set_sigma(self, sigma):
        ''' set sigma in pupil segmentation '''
        self.cfg.set('PUPILSEG','sigma',str(sigma))

    def get_sigma(self):
        ''' get sigma in pupil segmentation '''
        return c_float(self.cfg.getfloat('PUPILSEG','sigma'))



    # PUPILFIT section -----------

    def set_pupilfit_method(self, method):
        ''' set pupilfit method '''
        method_dict = {0:"ROBUST_LSQ", 1:"LSQ", 2:"RANSAC", 3:"RANSAC_SUPPORT"}
        self.cfg.set('PUPILFIT', 'method', method_dict[method])

    def get_pupilfit_method(self):
        ''' set pupilfit method '''
        method = self.cfg.get('PUPILFIT','method')
        method_dict = {0:"ROBUST_LSQ", 1:"LSQ", 2:"RANSAC", 3:"RANSAC_SUPPORT"}
        rev_method_dict = {y:x for x,y in method_dict.iteritems()}
        return c_int(rev_method_dict[method])


    def set_maxiterations(self,maxiterations):
        ''' set max iterations in pupil fit '''
        self.cfg.set('PUPILFIT','maxiterations',str(maxiterations))

    def get_maxiterations(self):
        ''' get mat iterations in pupil fit '''
        return c_int(self.cfg.getint('PUPILFIT','maxiterations'))


    def set_maxrefinements(self, maxrefinements):
        ''' set max refinements in pupil fit '''
        self.cfg.set('PUPILFIT','maxrefinements', str(maxrefinements))

    def get_maxrefinements(self):
        ''' get setting of max refinements in pupil fit '''
        return c_int(self.cfg.getint('PUPILFIT','maxrefinements'))


    def set_maxinlierperc(self,maxinlierperc):
        ''' set max percent of inliers in pupil fit '''
        self.cfg.set('PUPILFIT','maxinlierperc',str(maxinlierperc))

    def get_maxinlierperc(self):
        ''' get setting of max percent of inliers in pupil fit '''
        return c_float(self.cfg.getfloat('PUPILFIT','maxinlierperc'))


    
    # ARTIFACTS section ------------------
    
    def set_mrclean_enabled(self,enabled):
        ''' enable/disable MR artifact removal '''
        self.cfg.set('ARTIFACTS','mrclean', str(enabled))

    def get_mrclean_enabled(self):
        ''' get whether MR artifact removal is enabled '''
        return c_bool(self.cfg.getboolean('ARTIFACTS','mrclean'))


    def set_zthresh(self, zthresh):
        ''' set zthresh in artifact removal '''
        self.cfg.set('ARTIFACTS','zthresh',str(zthresh))

    def get_zthresh(self):
        ''' get zthresh in artifact removal '''
        return c_float(self.cfg.getfloat('ARTIFACTS','zthresh'))


    def set_motioncorr_method(self, method):
        ''' set motioncorr method '''
        method_dict = {0:"highpass", 1:"knownfixations"}
        self.cfg.set('ARTIFACTS', 'motioncorr', method_dict[method])

    def get_motioncorr_method(self):
        ''' set motioncorr method '''
        method = self.cfg.get('ARTIFACTS','motioncorr')
        method_dict = {0:"highpass", 1:"knownfixations"}
        rev_method_dict = {y:x for x,y in method_dict.iteritems()}
        return c_long(rev_method_dict[method])


    def set_mocokernel(self,mocokernel):
        ''' set mocokernel in ARTIFACTS '''
        self.cfg.set('ARTIFACTS','mocokernel',str(mocokernel))

    def get_mocokernel(self):
        ''' get mocokernel in ARTIFACTS '''
        return c_long(self.cfg.getint('ARTIFACTS','mocokernel'))



    # OUTPUT section ------------

    def set_graphics(self, enable):
        ''' enable/disable graphics output '''
        self.cfg.set('OUTPUT', 'graphics', str(enable))

    def get_graphics(self):
        ''' check whether graphics output is enabled '''
        return c_bool(self.cfg.getboolean('OUTPUT','graphics'))


    def set_verbose(self, enable):
        ''' enable/disable verbose output '''
        self.cfg.set('OUTPUT', 'verbose', str(enable))

    def get_verbose(self):
        ''' check whether verbose output is enabled '''
        return c_bool(self.cfg.getboolean('OUTPUT','verbose'))



    # add options to OSD 
    def create_atb_bar(self,pos):
        ''' create advanced tweak bar with setting for Mr. Gaze '''
        self.bar = atb.Bar(name = "Mr_Gaze_Detector", label="Mr. Gaze Controls",
            help="Mr. Gaze Params", color=(50, 50, 50), alpha=100,
            text='light', position=pos, refresh=.3, size=(200, 250))

        # VIDEO section 
        self.bar.add_var("downsampling", vtype=c_long, setter=self.set_downsampling, getter=self.get_downsampling, min=1, max=5)

        # PUPILDETECT section
        self.bar.add_var("pupil detect", vtype=c_bool, setter=self.set_pupildetect_enabled, getter=self.get_pupildetect_enabled)

        self.bar.add_var("specificity", vtype=c_long, setter=self.set_specificity, getter=self.get_specificity, min=0, max=100)

        self.bar.add_var("scalefactor", vtype=c_float, setter=self.set_scalefactor, getter=self.get_scalefactor, min=0, max=2, step=0.01)


        # PUPILSEG section
        self.bar.pupilseg_method_enum = atb.enum("pupilseg M",{"otsu":0,
                                                  "manual":1})
        self.bar.add_var("Pupilseg M", vtype=self.bar.pupilseg_method_enum, setter=self.set_pupilseg_method, getter=self.get_pupilseg_method, help="select pupil seg method")

        self.bar.add_var("pupildiameterperc", vtype=c_float, setter=self.set_pupildiameterperc, getter=self.get_pupildiameterperc, min=0, max=100)

        self.bar.add_var("glintdiameterperc", vtype=c_float, setter=self.set_glintdiameterperc, getter=self.get_glintdiameterperc, min=0, max=100)

        self.bar.add_var("pupilthresholdperc", vtype=c_float, setter=self.set_pupilthresholdperc, getter=self.get_pupilthresholdperc, min=0, max=100)

        self.bar.add_var("sigma", vtype=c_float, setter=self.set_sigma, getter=self.get_sigma, min=0, max=5)


        # PUPILFIT section -----------
        method_dict = {0:"ROBUST_LSQ", 1:"LSQ", 2:"RANSAC", 3:"RANSAC_SUPPORT"}
        rev_method_dict = {y:x for x,y in method_dict.iteritems()}
        self.bar.pupilfit_method_enum = atb.enum("pupilfit M", rev_method_dict)

        self.bar.add_var("Pupilfit M", vtype=self.bar.pupilfit_method_enum, setter=self.set_pupilfit_method, getter=self.get_pupilfit_method, help="select pupil fit method")

        self.bar.add_var("maxiterations", vtype=c_long, setter=self.set_maxiterations, getter=self.get_maxiterations, min=0, max=25)

        self.bar.add_var("maxrefinements", vtype=c_long, setter=self.set_maxrefinements, getter=self.get_maxrefinements, min=0, max=25)

        self.bar.add_var("maxinlierperc", vtype=c_float, setter=self.set_maxinlierperc, getter=self.get_maxinlierperc, min=0, max=100)



        # # ARTIFACTS section ------------------
        # self.bar.add_var("MR clean", vtype=c_bool, setter=self.set_mrclean_enabled, getter=self.get_mrclean_enabled)

        # self.bar.add_var("zthresh", vtype=c_float, setter=self.set_zthresh, getter=self.get_zthresh, min=0, max=25)

        # method_dict = {0:"highpass", 1:"knownfixations"}
        # rev_method_dict = {y:x for x,y in method_dict.iteritems()}
        # self.bar.motioncorr_method_enum = atb.enum("MoCo Method", rev_method_dict)
        # self.bar.add_var("Motioncorr M", vtype=self.bar.motioncorr_method_enum, setter=self.set_motioncorr_method, getter=self.get_motioncorr_method, help="select motion correction method")

        # self.bar.add_var("mocokernel", vtype=c_long, setter=self.set_mocokernel, getter=self.get_mocokernel, min=0, max=250)

        # OUTPUT section ------------
        self.bar.add_var("graphics", vtype=c_bool, setter=self.set_graphics, getter=self.get_graphics)

        self.bar.add_var("verbose", vtype=c_bool, setter=self.set_verbose, getter=self.get_verbose)


    def cleanup(self):
        config.SaveConfig(self.cfg, self.gpool.user_dir)
        return 
