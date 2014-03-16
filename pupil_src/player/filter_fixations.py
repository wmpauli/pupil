'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2014  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import cv2
from plugin import Plugin
import numpy as np
import atb
from ctypes import c_float, c_bool
from methods import denormalize,normalize
from player_methods import transparent_circle
import logging
from scan_path import Scan_Path

logger = logging.getLogger(__name__)

class Filter_Fixations(Plugin):
    """docstring
    This plugin classifies fixations and saccades by measuring dispersion and duration of gaze points 

    Methods of fixation detection are based on prior literature
    Saccade vs fixation assumptions are based on 
        (Salvucci & Goldberg, ETRA, 2000) http://www.cs.drexel.edu/~salvucci/publications/Salvucci-ETRA00.pdf
        (Evans et al, JEMR, 2012) http://www.jemr.org/online/5/2/6

    Smooth Pursuit/Ego-motion accounted for by optical flow in Scan Path plugin: 
        Reference literature (Kinsman et al. "Ego-motion compensation improves fixation detection in wearable eye tracking," ACM 2011)
    
    Fixations prior knowledge from literature review
        + Fixations rarely less than 100ms duration
        + Fixations between 200-400ms in duration
        + dispersion = how much movement is allowed within one fixation (e.g. > 8 pixels movement is no longer fixation)
        + duration = how long must recent_pupil_positions remain within dispersion threshold before classified as fixation (e.g. at least 100ms)

    Overview Diagram
        + Scan path supplies a window into the past set by user (must be >= 0.4s)
        + The sample/anchor point is taken approx 0.2s away from most current timestamp
        + Cuttoff is 0.4 seconds in the past = theoretical maximum duration of fixation 

        past[         scan path history         ]now
            [- - - - - - - - - <-------s------->] 
                                       s-- t -->
                         cutoff<-- t --s
                               <  max fixation >

        + Preliminary classification candidates/support if within distance threshold of sample using manhattan distance
                  dx  
                +---pt
                |  /
             dy | /
                |/
                s

        + Final classification of sample as fixation if supporting candidates & sample within min_duration threshold
            + Check for min_duration 0.1s in sliding window around sample (including sample)

                |0.1s|
            <---|--s-|---->
            <-----|s---|-->
            <--|---s|-----> 

            - if fixations are >= 0.1s and inclusive of sample and within distance threshold, then sample classified as fixations

            <--|**s**|---> == fixation
    """
    def __init__(self, g_pool=None,distance=8.0,show_saccades=False,gui_settings={'pos':(10,470),'size':(300,100),'iconified':False}):
        super(Filter_Fixations, self).__init__()

        self.g_pool = g_pool
        # let the plugin work after most other plugins
        self.order = .7

        # user settings
        self.distance = c_float(float(distance))
        self.show_saccades = c_bool(bool(show_saccades))
        self.min_duration = 0.10
        self.max_duration = 0.40
        self.gui_settings = gui_settings

        self.sp_active = True

        # algorithm working data
        self.d = {}
        self.sample_pt = None
        self.past_pt = None
        self.present_pt = None
        self.candidates = []
        '''
        d[p["timestamp"]] = "fixation"
        p["timestamp"]
        p["type"] = d[p"timestamp"]
        '''

    def update(self,frame,recent_pupil_positions,events):
        img = frame.img
        img_shape = img.shape[:-1][::-1] # width,height

        # initialize Scan Path so we can use its history and optical flow
        if any(isinstance(p,Scan_Path) for p in self.g_pool.plugins):
            if self.sp_active:
                pass
            else:
                self.set_bar_ok(True)
                self.sp_active = True
        else:
            if self.sp_active:
                self.set_bar_ok(False)
                self.sp_active = False
            else:
                pass

        try:
            self.present_pt = recent_pupil_positions[-1]
            cutoff = self.present_pt['timestamp'] - self.max_duration
            self.candidates = [g for g in recent_pupil_positions if g['timestamp']>cutoff]
            self.past_pt = self.candidates[0]
            dt = self.present_pt['timestamp']-self.past_pt['timestamp']
            
            if dt < 0.1:
                # no chance of there being a fixation here anyways
                self.candidates = []
                
                logger.debug("not enough samples for classification - dt: %03f" %(dt))
            else:
                t = self.present_pt['timestamp']- self.max_duration*0.5
                self.sample_pt = min(self.candidates, key=lambda k: abs(k['timestamp']-t))
                # remove sample point from candidate list
                # self.candidates[:] = [p for p in self.candidates if p['timestamp'] != self.sample_pt['timestamp']]
                
                logger.debug("cutoff: %3f\tcandidates: %s" %(cutoff, len(self.candidates)))
                logger.debug("past_pt: %03f\t sample_pt %03f\t present_pt: %03f" %(self.past_pt['timestamp'], self.sample_pt['timestamp'], self.present_pt['timestamp']))
        except:
            # no recent_pupil_positions
            pass

        # classify sample point fixation or saccade
        if self.candidates and self.sample_pt:
            for p in self.candidates:
                if self.manhattan_dist_denormalize(self.sample_pt, p, img_shape) < self.distance.value:
                    p['support'] = True
                else:
                    p['support'] = False
                logger.debug("%s @ %03f" %(p['support'],p['timestamp']))

            
            min_fix = min([p['timestamp'] for p in self.candidates if p['support']])
            max_fix = max([p['timestamp'] for p in self.candidates if p['support']])
            dt = max_fix-min_fix
            logger.debug("min_fix: %03f\t max_fix: %03f\tdt: %s" %(min_fix,max_fix,dt))
            
            if dt > self.min_duration and min_fix <= self.sample_pt['timestamp'] <= max_fix:
                self.d[self.sample_pt['timestamp']] = "fixation"

            # draw fixations
            # inject knowledge of now and knowledge of 'past'
            # p["type"] = d[p"timestamp"]
            # if self.show_saccades.value:
            #     pts = [denormalize(pt['norm_gaze'],frame.img.shape[:-1][::-1],flip_y=True) for pt in saccades if pt['norm_gaze'] is not None]
            #     for pt in pts:
            #         transparent_circle(frame.img, pt, radius=20, color=(255,150,0,100), thickness=2)


    def init_gui(self,pos=None):
        import atb
        pos = self.gui_settings['pos']
        atb_label = "Filter Fixations"
        self._bar = atb.Bar(name =self.__class__.__name__+str(id(self)), label=atb_label,
            help="polyline", color=(50, 50, 50), alpha=50,
            text='light', position=pos,refresh=.1, size=self.gui_settings['size'])

        self._bar.iconified = self.gui_settings['iconified']
        self._bar.add_var('distance in pixels',self.distance,min=0,step=0.1)
        self._bar.add_var('show saccades',self.show_saccades)
        self._bar.add_button('remove',self.unset_alive)



    def set_bar_ok(self,ok):
        if ok:
            self._bar.color = (50, 50, 50)
            self._bar.label = "Filter Fixations"
        else:
            self._bar.color = (250, 50, 50)
            self._bar.label = "Filter Fixations: Turn on Scan_Path!"

    def unset_alive(self):
        self.alive = False


    def get_init_dict(self):
        d = {'distance':self.distance.value, 'show_saccades':self.show_saccades.value}
        if hasattr(self,'_bar'):
            gui_settings = {'pos':self._bar.position,'size':self._bar.size,'iconified':self._bar.iconified}
            d['gui_settings'] = gui_settings
        return d


    def clone(self):
        return Filter_Fixations(**self.get_init_dict())


    def cleanup(self):
        """ called when the plugin gets terminated.
        This happends either voluntary or forced.
        if you have an atb bar or glfw window destroy it here.
        """
        self._bar.destroy()

    def manhattan_dist_denormalize(self, gp1, gp2, img_shape):
        gp1_norm = denormalize(gp1['norm_gaze'], img_shape,flip_y=True)
        gp2_norm = denormalize(gp2['norm_gaze'], img_shape,flip_y=True)
        x_dist =  abs(gp1_norm[0] - gp2_norm[0])
        y_dist = abs(gp1_norm[1] - gp2_norm[1])
        man = x_dist + y_dist
        return man

