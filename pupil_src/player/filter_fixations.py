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
from ctypes import c_float
from methods import denormalize,normalize
from player_methods import transparent_circle
import logging
from scan_path import Scan_Path

logger = logging.getLogger(__name__)

class Filter_Fixations(Plugin):
    """docstring
    This plugin detects fixations by measuring dispersion and duration between recent_pupil_positions
    this allows one to effectively filter out saccades

    Methods of fixation detection are based on prior literature
    
    Saccade vs fixation assumptions are based on 
        (Salvucci & Goldberg, ETRA, 2000) http://www.cs.drexel.edu/~salvucci/publications/Salvucci-ETRA00.pdf
        (Evans et al, JEMR, 2012) http://www.jemr.org/online/5/2/6

    Fixations notes/assumptions from literature
        + Fixations rarely less than 100ms duration
        + Fixations between 200-400ms in duration
        + Fixations (as word implies) are when the eye is not moving (or within a tolerance of movement)     

    Fixation thresholds:
        + dispersion = how much movement is allowed within one fixation (e.g. > 8 pixels movement is no longer fixation)
        + duration = how long must recent_pupil_positions remain within dispersion threshold before classified as fixation (e.g. at least 100ms)

    Smooth Pursuit/Ego-motion (additional plugin to be run prior to fixations): 
        + VOR - when moving the head while fixating on an object we need to compensate for scene/world movement
        + We compensate by using optical flow - sticking gaze points onto pixels in the scene in the Scan Path 
        Reference literature (Kinsman et al. "Ego-motion compensation improves fixation detection in wearable eye tracking," ACM 2011)

    Overview:
        + Get recent_pupil_positions from Scan Path plugin
            + recent_pupil_positions gaze points "stuck" to the pixel to compensate for Ego-motion
            + timeframe of Scan Path is the temporal window within which we calculate/determine fixations
        + Use Sliding Filter to classify gaze point as fixation or saccade
            + get gaze point at the middle of recent_pupil_positions
            + create sub time slice of 400ms (200ms past, 200ms future)
            + check distance between gaze point vs past & future
            + check that time >= 0.1 second (approx. 3 frames)
    """
    def __init__(self, g_pool=None,distance=8.0,max_duration=400.0,gui_settings={'pos':(10,470),'size':(300,100),'iconified':False}):
        super(Filter_Fixations, self).__init__()

        self.g_pool = g_pool
        # let the plugin work after most other plugins
        self.order = .7

        # user settings
        self.distance = c_float(float(distance))
        self.max_duration = c_float(float(max_duration))
        self.min_duration = 0.10
        self.gui_settings = gui_settings

        self.sp_active = True

    def update(self,frame,recent_pupil_positions,events):

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


        img = frame.img
        img_shape = img.shape[:-1][::-1] # width,height

        if recent_pupil_positions:
            past_gp = recent_pupil_positions[:len(recent_pupil_positions)/2]
            curr_gp = recent_pupil_positions[len(recent_pupil_positions)/2]
            future_gp = recent_pupil_positions[(len(recent_pupil_positions)/2)+1:]

            now = curr_gp['timestamp']
            past_cutoff = now-0.2 # 200 ms window/2 = max saccade duration
            future_cutoff = now+0.2 # 

            past_gp = [g for g in past_gp if g['timestamp']>=past_cutoff and self.manhattan_dist_denormalize(curr_gp, g, img_shape) < self.distance.value]
            future_gp = [g for g in future_gp if g['timestamp']<=future_cutoff and self.manhattan_dist_denormalize(curr_gp, g, img_shape) < self.distance.value] 
            fixation_candidates = past_gp + future_gp

            # if we detected a potential fixation, was it longer than 100 milliseconds?
            future_t, past_t = False, False
            if past_gp:
                past_gp.sort(key=lambda x: x['timestamp'])
                past_t = now-past_gp[0]['timestamp']
                print "past_t: ", past_t

            if future_gp:
                future_gp.sort(key=lambda x: x['timestamp'])
                future_t = future_gp[-1]['timestamp']-now

            if future_t and past_t:
                if future_t+past_t < 0.1:
                    print "not a probable fixation..."
                    print "fixation_time: ", future_t+past_t

            # print "future_gp: ", future_gp

            recent_pupil_positions[:] = fixation_candidates[:]
            recent_pupil_positions.sort(key=lambda x: x['timestamp']) #this may be redundant...



    def init_gui(self,pos=None):
        import atb
        pos = self.gui_settings['pos']
        atb_label = "Filter Fixations"
        self._bar = atb.Bar(name =self.__class__.__name__+str(id(self)), label=atb_label,
            help="polyline", color=(50, 50, 50), alpha=50,
            text='light', position=pos,refresh=.1, size=self.gui_settings['size'])

        self._bar.iconified = self.gui_settings['iconified']
        self._bar.add_var('distance in pixels',self.distance,min=0,step=0.1)
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
        d = {'distance':self.distance.value, 'max_duration':self.max_duration.value}
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

