'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2014  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import os
from time import time, sleep
from file_methods import Persistent_Dict
import logging
from ctypes import c_int,c_bool,c_float
import numpy as np
import atb
from glfw import *
from gl_utils import basic_gl_setup,adjust_gl_view, clear_gl_screen, draw_gl_point_norm,make_coord_system_pixel_based,make_coord_system_norm_based,create_named_texture,draw_named_texture,draw_gl_polyline
from methods import *
from uvc_capture import autoCreateCapture, FileCaptureError, EndofVideoFileError, CameraCaptureError
from calibrate import get_map_from_cloud
from pupil_detectors import Canny_Detector, MSER_Detector, Blob_Detector, MrGaze_Detector

def eye(g_pool,cap_src,cap_size):
    """
    Creates a window, gl context.
    Grabs images from a capture.
    Streams Pupil coordinates into g_pool.pupil_queue
    """

    # modify the root logger for this process
    logger = logging.getLogger()
    # remove inherited handlers
    logger.handlers = []
    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(g_pool.user_dir,'eye.log'),mode='w')
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('EYE Process: %(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    formatter = logging.Formatter('E Y E Process [%(levelname)s] %(name)s : %(message)s')
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    # create logger for the context of this function
    logger = logging.getLogger(__name__)
#    border = 300

    # Callback functions
    def on_resize(window,w, h):
        adjust_gl_view(w,h,window)
        norm_size = normalize((w,h),glfwGetWindowSize(window))
        fb_size = denormalize(norm_size,glfwGetFramebufferSize(window))
        atb.TwWindowSize(*map(int,fb_size))


    def on_key(window, key, scancode, action, mods):
        if not atb.TwEventKeyboardGLFW(key,int(action == GLFW_PRESS)):
            if action == GLFW_PRESS:
                if key == GLFW_KEY_ESCAPE:
                    on_close(window)

    def on_char(window,char):
        if not atb.TwEventCharGLFW(char,1):
            pass

    def on_button(window, button, action, mods):
        if not atb.TwEventMouseButtonGLFW(button,int(action == GLFW_PRESS)):
            if action == GLFW_PRESS: # mouse button was pressed
                if bar.display.value == 1:
                    pos = glfwGetCursorPos(window)
                    pos = normalize(pos,glfwGetWindowSize(window))
                    pos = denormalize(pos,(frame.img.shape[1],frame.img.shape[0]) ) # pos in frame.img pixels
                    if bar.draw_roi.value:
                        u_r.setStart(pos)
                        bar.drawing_roi.value = 1
                    elif bar.pick_pupil.value:
                        print "pupil location is: " + str(pos)
            else: # button released
                if bar.drawing_roi.value == 1:
                    bar.drawing_roi.value = 0
                    # now also set the AOI for the camera
                    # roi_tmp = u_r.get()
                    # try: 
                    #     if hasattr(cap, 'capture'):
                    #         cap.capture.set_size(x = roi_tmp[0], y = roi_tmp[1], width=roi_tmp[2] - roi_tmp[0], height=roi_tmp[3] - roi_tmp[1])
                    # except AttributeError:
                    #     pass

    def reset_aoi():
        u_r.setStart((0,0))
        u_r.setEnd((frame.img.shape[1], frame.img.shape[0]))
        # cap.capture.set_size()

    def on_pos(window,x, y):
        ''' seems to get called if mouse is pressed and moved ''' 
        norm_pos = normalize((x,y),glfwGetWindowSize(window))
        fb_x,fb_y = denormalize(norm_pos,glfwGetFramebufferSize(window))
        if atb.TwMouseMotion(int(fb_x),int(fb_y)):
            pass

        if bar.drawing_roi.value == 1:
            pos = denormalize(norm_pos,(frame.img.shape[1],frame.img.shape[0]) ) # pos in frame.img pixels
            u_r.setEnd(pos)

    def on_scroll(window,x,y):
        if not atb.TwMouseWheel(int(x)):
            pass

    def on_close(window):
        g_pool.quit.value = True
        logger.info('Process closing from window')


    # Helper functions called by the main atb bar
    def start_roi():
        bar.display.value = 1
        bar.drawing_roi.value = 2

    def update_fps():
        old_time, bar.timestamp = bar.timestamp, time()
        dt = bar.timestamp - old_time
        if dt:
            bar.fps.value += .05 * (1. / dt - bar.fps.value)
            bar.dt.value = dt

    def get_from_data(data):
        """
        helper for atb getter and setter use
        """
        return data.value


    # load session persistent settings
    session_settings = Persistent_Dict(os.path.join(g_pool.user_dir,'user_settings_eye') )
    def load(var_name,default):
        return session_settings.get(var_name,default)
    def save(var_name,var):
        session_settings[var_name] = var

    # Initialize capture
    cap = autoCreateCapture(cap_src, cap_size,timebase=g_pool.timebase)

    if cap is None:
        logger.error("Did not receive valid Capture")
        return
    # check if it works
    frame = cap.get_frame()
    #    frame.img = frame.img[border:frame.img.shape[0] - border,border:frame.img.shape[1] - border]

    if frame.img is None:
        logger.error("Could not retrieve image from capture")
        cap.close()
        return

    u_r = Roi(frame.img.shape)
    u_r.set(load('roi',default=None))

    # roi_tmp = u_r.get()
    # if hasattr(cap, 'capture'):
    #     cap.capture.set_size(x = roi_tmp[0], y = roi_tmp[1], width = roi_tmp[2] - roi_tmp[0], height = roi_tmp[3] - roi_tmp[1])

#    border = 100
#    frame.img = frame.img[u_r.lY-border:u_r.uY+border,u_r.lX-border:u_r.uX+border]
    height, width = frame.img.shape[:2]

    writer = None

    pupil_detector = MrGaze_Detector(g_pool)

    atb.init()
    # Create main ATB Controls
    bar = atb.Bar(name = "Eye", label="Display",
            help="Scene controls", color=(50, 50, 50), alpha=100,
            text='light', position=(10, 10),refresh=.3, size=(200, 150))
    bar.fps = c_float(0.0)
    bar.fps_et = c_float(23.0)
    bar.skip_frames = c_int(0)
    bar.timestamp = time()
    bar.dt = c_float(0.0)
    bar.sleep = c_float(0.0)
    bar.display = c_int(load('bar.display',0))
    bar.draw_pupil = c_bool(load('bar.draw_pupil',True))
    bar.drawing_roi = c_int(0)
    bar.draw_roi = c_bool(load('bar.draw_roi',True))
#    bar.pick_pupil = c_bool(load('bar.draw_roi',False))

    dispay_mode_enum = atb.enum("Mode",{"Camera Image":0,
                                        "Region of Interest":1,
                                        "Algorithm":2,
                                        "CPU Save": 3})

    bar.add_var("FPS (rec)", bar.fps, step=1., readonly=True)
    bar.add_var("FPS (ET)", bar.fps_et, step=1, min=1, readonly=False)
    bar.add_var("Mode", bar.display,vtype=dispay_mode_enum, help="select the view-mode")
    bar.add_var("Show_Pupil_Point", bar.draw_pupil)
    bar.add_var("Draw ROI", bar.draw_roi)
    bar.add_button("reset AOI", reset_aoi, help="Click to reset AOI.")

    bar.add_var("SlowDown",bar.sleep, step=0.01,min=0.0)
    bar.add_var("SaveSettings&Exit", g_pool.quit)
    
    # create atb bar for camera
    # cap.create_atb_bar(pos=(220,10))

    # create a bar for the detector
    pupil_detector.create_atb_bar(pos=(10,160))


    glfwInit()
    window = glfwCreateWindow(width, height, "Eye", None, None)
    glfwMakeContextCurrent(window)

    # Register callbacks window
    glfwSetWindowSizeCallback(window,on_resize)
    glfwSetWindowCloseCallback(window,on_close)
    glfwSetKeyCallback(window,on_key)
    glfwSetCharCallback(window,on_char)
    glfwSetMouseButtonCallback(window,on_button)
    glfwSetCursorPosCallback(window,on_pos)
    glfwSetScrollCallback(window,on_scroll)

    glfwSetWindowPos(window,800,0)
    on_resize(window,width,height)

    # gl_state settings
    basic_gl_setup()
    g_pool.image_tex = create_named_texture(frame.img)

    # refresh speed settings
    glfwSwapInterval(0)

    lpft = 0.1 # last time a frame was processed by pupil_dector
    # ldu = 0.1 # last time display has been updated
    # skipped_frames = 0
    # event loop
    border = 0
    skipped_frames = 0
    while not g_pool.quit.value:
        # Get an image from the grabber
        try:
            frame = cap.get_frame()
#            frame.img = frame.img[u_r.lY-border:u_r.uY+border,u_r.lX-border:u_r.uX+border]
#            frame.img = frame.img[border:frame.img.shape[0] - border,border:frame.img.shape[1] - border]
        except CameraCaptureError:
            logger.error("Capture from Camera Failed. Stopping.")
            break
        except EndofVideoFileError:
            logger.warning("Video File is done. Stopping")
            break

        update_fps()
        sleep(bar.sleep.value) # for debugging only


        ###  RECORDING of Eye Video (on demand) ###
        # Setup variables and lists for recording
        if g_pool.eye_rx.poll():
            command = g_pool.eye_rx.recv()
            if command is not None:
                record_path = command
                logger.info("Will save eye video to: %s"%record_path)
                video_path = os.path.join(record_path, "eye.avi")
                timestamps_path = os.path.join(record_path, "eye_timestamps.npy")

                fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
                writer = cv2.VideoWriter(video_path, fourcc, bar.fps.value, (abs(u_r.uX-u_r.lX), abs(u_r.uY - u_r.lY)))
                timestamps = []
            else:
                logger.info("Done recording eye.")
                writer = None
                np.save(timestamps_path,np.asarray(timestamps))
                del timestamps

        if writer:
            pupil_img = frame.img[u_r.lY:u_r.uY,u_r.lX:u_r.uX]
            writer.write(pupil_img)
            timestamps.append(frame.timestamp)


        # pupil ellipse detection
        process_frame = False
        if time() - lpft > 1.0/bar.fps_et.value:
            process_frame = True
            result = pupil_detector.detect(frame,user_roi=u_r,visualize=bar.display.value == 2)
            lpft = time()
        
            # stream the result
            g_pool.pupil_queue.put(result)

        # VISUALIZATION direct visualizations on the frame.img data
        if bar.display.value == 1:
            # and a solid (white) frame around the user defined ROI
            r_img = frame.img[u_r.lY:u_r.uY,u_r.lX:u_r.uX]
            r_img[:,0] = 255,255,255
            r_img[:,-1]= 255,255,255
            r_img[0,:] = 255,255,255
            r_img[-1,:]= 255,255,255



        # GL-drawing, only draw at specified rate
        if process_frame: #bar.skip_frames.value == 0 or skipped_frames == bar.skip_frames.value:
            skipped_frames = 0
            clear_gl_screen()
            make_coord_system_norm_based()
            if bar.display.value != 3:
                draw_named_texture(g_pool.image_tex,frame.img)
            else:
                draw_named_texture(g_pool.image_tex)
            make_coord_system_pixel_based(frame.img.shape)


            if result['norm_pupil'] is not None and bar.draw_pupil.value:
                if result.has_key('axes'):
                    pts = cv2.ellipse2Poly( (int(result['center'][0] + 1),int(result['center'][1] + 1)),
                                            (int(result["axes"][0]/2),int(result["axes"][1]/2)),
                                            int(result["angle"]),0,360,15)
                    draw_gl_polyline(pts,(1.,0,0,.5))
                draw_gl_point_norm(result['norm_pupil'],color=(1.,0.,0.,0.5))

            atb.draw()
            glfwSwapBuffers(window)
            glfwPollEvents()
        else:
            skipped_frames += 1

    # END while running

    # in case eye reconding was still runnnig: Save&close
    if writer:
        logger.info("Done recording eye.")
        writer = None
        np.save(timestamps_path,np.asarray(timestamps))


    # save session persistent settings
    save('roi',u_r.get())
    save('bar.display',bar.display.value)
    save('bar.draw_pupil',bar.draw_pupil.value)
    session_settings.close()

    pupil_detector.cleanup()
    cap.close()
    atb.terminate()
    glfwDestroyWindow(window)
    glfwTerminate()

    #flushing queue incase world process did not exit gracefully
    while not g_pool.pupil_queue.empty():
        g_pool.pupil_queue.get()
    g_pool.pupil_queue.close()

    logger.debug("Process done")

def eye_profiled(g_pool,cap_src,cap_size):
    import cProfile,subprocess,os
    from eye import eye
    cProfile.runctx("eye(g_pool,cap_src,cap_size)",{"g_pool":g_pool,'cap_src':cap_src,'cap_size':cap_size},locals(),"eye.pstats")
    loc = os.path.abspath(__file__).rsplit('pupil_src', 1)
    gprof2dot_loc = os.path.join(loc[0], 'pupil_src', 'shared_modules','gprof2dot.py')
    subprocess.call("python "+gprof2dot_loc+" -f pstats eye.pstats | dot -Tpng -o eye_cpu_time.png", shell=True)
    print "created cpu time graph for eye process. Please check out the png next to the eye.py file"

