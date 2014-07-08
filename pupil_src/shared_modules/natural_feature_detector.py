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
from collections import namedtuple

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

FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH    = 6
flann_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2

MIN_MATCH_COUNT = 10

ReferenceSurface = namedtuple('ReferenceSurface', 'image, keypoints, descrs, data')
TrackedReference = namedtuple('TrackedReference', 'reference, p0, p1, H, quad')

class Natural_Feature_Detector(Plugin):
    """docstring
    """
    def __init__(self,g_pool,size=2,color=(0.,1.0,0.5,.5),atb_pos=(320,200)):
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

        # setup the matcher
        self.matcher = cv2.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)

        # features detected 
        self.reference_surfaces = []
        self.add_reference_surface("test_003.png")

        self.tracked = []

    def add_reference_surface(self, file_name):
        img = cv2.imread(os.path.join(self.g_pool.rec_dir, file_name))
        k = self.detector.detect(img, None)
        key_points, descriptors = self.extractor.compute(img, k)
        
        print "number of kp in ref: ", len(key_points)
        self.matcher.add([descriptors])

        ref_surface = ReferenceSurface(image=img, keypoints=key_points, descrs=descriptors, data=None)
        self.reference_surfaces.append(ref_surface)


    def match(self):

        # find match between ref and current frame
        matches = self.matcher.knnMatch(self.frame_descrs, k=2)

        matches = [m[0] for m in matches if len(m) == 2 and m[0].distance < m[1].distance * 0.75]
        if len(matches) < MIN_MATCH_COUNT:
            return []

        matches_by_id = [[] for _ in xrange(len(self.reference_surfaces))]
        
        print "number of potential matches: ", len(matches)

        for m in matches:
            matches_by_id[m.imgIdx].append(m)

        tracked = []
        for imgIdx, matches in enumerate(matches_by_id):
            if len(matches) < MIN_MATCH_COUNT:
                print "early exit 1"
                continue
            ref = self.reference_surfaces[imgIdx]
            p0 = [ref.keypoints[m.trainIdx].pt for m in matches]
            p1 = [self.frame_kps[m.queryIdx].pt for m in matches]
            p0, p1 = np.float32((p0, p1))
            H, status = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)
            status = status.ravel() != 0
            if status.sum() < MIN_MATCH_COUNT:
                print "status.sum: ", status.sum()
                continue            
            p0, p1 = p0[status], p1[status]

            track = TrackedReference(reference=ref, p0=p0, p1=p1, H=None, quad=None)
            tracked.append(track)
        tracked.sort(key = lambda t: len(t.p0), reverse=True)            
        return tracked

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
        # detect the features and extract descriptors per frame
        # kp, descrs = self.detect_features(frame.img)
        frame_kp = self.detector.detect(frame.img, None)
        print "number of kp in frame: ", len(frame_kp)
        if len(frame_kp) < MIN_MATCH_COUNT: 
            return

        self.frame_kps, self.frame_descrs = self.extractor.compute(frame.img, frame_kp)        

        self.tracked = self.match()



    def gl_display(self):
        """
        Display marker and surface info inside world screen
        """
        size = self.size.value
        color = self.color[::]
        pts = [ref.p1 for ref in self.tracked]

        draw_gl_points(pts,size=10,color=color)



    def cleanup(self):
        """ called when the plugin gets terminated.
        This happends either voluntary or forced.
        if you have an atb bar or glfw window destroy it here.
        """
        self._bar.destroy()










