import logging
import os
from glob import glob

import numpy as np
from pyglui import ui
from PIL import Image, ImageDraw

from plugin import Plugin
from methods import  normalize
from file_methods import Persistent_Dict, save_object
from video_capture import autoCreateCapture, EndofVideoFileError, CameraCaptureError
from circle_detector import find_concetric_circles
from calibration_routines import calibration_plugins, gaze_mapping_plugins
from plugin import Plugin_List

logger = logging.getLogger(__name__)

def correlate_data(data,timestamps):
    '''
    data:  list of data :
        each datum is a dict with at least:
            timestamp: float

    timestamps: timestamps list to correlate  data to

    this takes a data list and a timestamps list and makes a new list
    with the length of the number of timestamps.
    Each slot contains a list that will have 0, 1 or more assosiated data points.

    Finally we add an index field to the datum with the associated index
    '''
    timestamps = list(timestamps)
    data_by_frame = [[] for i in timestamps]

    frame_idx = 0
    data_index = 0

    data.sort(key=lambda d: d['timestamp'])

    while True:
        try:
            datum = data[data_index]
            # we can take the midpoint between two frames in time: More appropriate for SW timestamps
            ts = ( timestamps[frame_idx]+timestamps[frame_idx+1] ) / 2.
            # or the time of the next frame: More appropriate for Sart Of Exposure Timestamps (HW timestamps).
            # ts = timestamps[frame_idx+1]
        except IndexError:
            # we might loose a data point at the end but we dont care
            break

        if datum['timestamp'] <= ts:
            datum['index'] = frame_idx
            data_by_frame[frame_idx].append(datum)
            data_index +=1
        else:
            frame_idx+=1

    return data_by_frame

class split_into_frames(Plugin):
    """
    transfer undetected pupil frames to crowdeye api
    """

    def __init__(self,g_pool):
        super(split_into_frames, self).__init__(g_pool)
        self.menu = None
        self.sub_menu = None
        self.buttons = []
        # self.new_annotation_hotkey = 'e'

        self.current_frame = -1

        print 'split_into_frames_init'
        self.order = .9
        self.active = False
        self.total_frames = 0
        self.rec_path = ''
        self.g_pool = g_pool
        self.active_cal = False


    def gaze_mapper(self, pupil_positions):
        gaze_pts = []
        pupil_confidence_threshold = 0.6
        for p in pupil_positions:
            if p['confidence'] > self.g_pool.pupil_confidence_threshold:
                norm_pos = normalize()
                gaze_point = self.map_fn(p['norm_pos'])
                gaze_pts.append({'norm_pos':gaze_point,'confidence':p['confidence'],'timestamp':p['timestamp'],'base':[p]})

        # events['gaze_positions'] = gaze_pts

    def save_image(self, img, save_to_file, ellipse=None, center=None):
        new_img = Image.fromarray(img)
        draw = ImageDraw.Draw(new_img)
        if ellipse is not None:
            center = (int(ellipse['center'][0]),int(ellipse['center'][1]))
            axes = (int(ellipse['axes'][0]/2),int(ellipse['axes'][1]/2))
            draw.ellipse((center[0] - axes[0], center[1] - axes[1], center[0] + axes[0], center[1] + axes[1]), fill = None, outline =255)
        if center and not ellipse:
            diameter = 10
            draw.ellipse((center[0] - diameter, center[1] - diameter, center[0] + diameter, center[1] + diameter), fill=(202,255,90), outline =255)

        del draw
        new_img.save(save_to_file)

    def get_pupil_list(self, data_path, settings_path):
        # Pupil detectors
        from pupil_detectors import Detector_2D, Detector_3D
        pupil_detectors = {Detector_2D.__name__:Detector_2D,Detector_3D.__name__:Detector_3D}
        # get latest settings
        session_settings = Persistent_Dict(settings_path)
        pupil_detector_settings = session_settings.get('pupil_detector_settings',None)
        last_pupil_detector = pupil_detectors[session_settings.get('last_pupil_detector',Detector_2D.__name__)]
        pupil_detector = last_pupil_detector(self.g_pool,pupil_detector_settings)

        # Detect pupil
        video_path = [f for f in glob(os.path.join(data_path,"eye0.*")) if f[-3:] in ('mp4','mkv','avi')][0]
        # video_capture = cv2.VideoCapture(e_video_path)
        # pos_frame = video_capture.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)

        from ctypes import c_double
        from multiprocessing import Value
        timebase = Value(c_double,0)
        video_capture = autoCreateCapture(video_path, timebase=timebase)
        default_settings = {'frame_size':(640,480),'frame_rate':30}
        video_capture.settings = session_settings.get('capture_settings',default_settings)
        # previous_settings = session_settings.get('capture_settings',default_settings)
        # if previous_settings and previous_settings['name'] == cap.name:
        #     video_capture.settings = previous_settings
        # else:
        #     video_capture.settings = default_settings

        frame = None
        # Test capture
        sequence = 0
        try:
            frame = video_capture.get_frame()
            sequence += 1
        except CameraCaptureError:
            logger.error("Could not retrieve image from world.mp4")
            video_capture.close()
            return

        from player_settings.plugins.offline_crowd_process.ui_roi import UIRoi
        self.g_pool.display_mode = session_settings.get('display_mode','camera_image')
        self.g_pool.display_mode_info_text = {'camera_image': "Raw eye camera image. This uses the least amount of CPU power",
                                'roi': "Click and drag on the blue circles to adjust the region of interest. The region should be as small as possible, but large enough to capture all pupil movements.",
                                'algorithm': "Algorithm display mode overlays a visualization of the pupil detection parameters on top of the eye video. Adjust parameters within the Pupil Detection menu below."}
        self.g_pool.u_r = UIRoi(frame.img.shape)
        self.g_pool.u_r.set(session_settings.get('roi',self.g_pool.u_r.get()))
        save_undetected = os.path.join(data_path,'undetected')
        save_detected = os.path.join(data_path,'detected')
        if not os.path.exists(save_detected):
            print "creating %s and save images to"%save_detected
            os.makedirs(save_detected)
        if not os.path.exists(save_undetected):
            print "creating %s and save images to"%save_undetected
            os.makedirs(save_undetected)
        pupil_list = []
        while True:
            try:
                # get frame by frame
                frame = video_capture.get_frame()
                # pupil ellipse detection
                result = pupil_detector.detect(frame, self.g_pool.u_r, self.g_pool.display_mode == 'algorithm')
                result['id'] = 0
                sequence +=1
                # Use sequence to sort the frames on server and when back from the server
                result['seq'] = sequence
                pupil_list += [result]
                if result['confidence'] >0:
                    if result.has_key('ellipse'):
                        self.save_image(frame.img, os.path.join(save_detected,'%s_%s.jpg'%(repr(result['timestamp']),
                                                                       repr(result['confidence']))), result['ellipse'])
                else:
                    self.save_image(frame.img, os.path.join(save_undetected,'%s_%s.jpg'%(repr(result['timestamp']),
                                                                       repr(result['confidence']))))
            except EndofVideoFileError:
                logger.warning("Eye video file is done. Stopping")
                break
        return pupil_list

    def detect_marker(self, frame):
        gray_img  = frame.gray
        markers = find_concetric_circles(gray_img,min_ring_count=3)
        detected = False
        pos = None
        if len(markers) > 0:
            detected = True
            marker_pos = markers[0][0][0] #first marker innermost ellipse, pos
            pos = marker_pos # normalize(marker_pos,(frame.width,frame.height),flip_y=True)
        else:
            detected = False
            pos = None #indicate that no reference is detected
        ''' return either detected and pos or just detected
        If I'm to crowdsource all of this then I just need to know when is the stop marker was detected
        '''
        # # center dark or white?
        # if detected:
        #     second_ellipse =  markers[0][1]
        #     col_slice = int(second_ellipse[0][0]-second_ellipse[1][0]/2),int(second_ellipse[0][0]+second_ellipse[1][0]/2)
        #     row_slice = int(second_ellipse[0][1]-second_ellipse[1][1]/2),int(second_ellipse[0][1]+second_ellipse[1][1]/2)
        #     marker_gray = gray_img[slice(*row_slice),slice(*col_slice)]
        #     avg = cv2.mean(marker_gray)[0] #CV2 fn return has changed!
        #     center = marker_gray[second_ellipse[1][1]/2,second_ellipse[1][0]/2]
        #     rel_shade = center-avg
        #
        #     #auto_stop logic
        #     if rel_shade > 30:
        #         #bright marker center found
        #         auto_stop +=1
        #         stop_marker_found = True
        #         active_cal = False
        #     else:
        #         auto_stop = 0
        #         stop_marker_found = False
        return detected, pos

    def split_recordings(self):
        data_path = '/Developments/NCLUni/pupil_crowd4Jul16/recordings/2016_07_06/003'#self.rec_path
        # Set user_dir to data_path so all related plugins save to the same folder as the recordings
        self.g_pool.user_dir = data_path
        # Manage plugins
        plugin_by_index =  calibration_plugins+gaze_mapping_plugins
        name_by_index = [p.__name__ for p in plugin_by_index]
        plugin_by_name = dict(zip(name_by_index,plugin_by_index))

        settings_path = os.path.join(data_path[:data_path.index('recordings')] + 'capture_settings')

        # Step 1: when possible detect all pupil positions
        pupil_list = self.get_pupil_list(data_path, os.path.join(settings_path,'user_settings_eye0'))

        if pupil_list:
            # create events variable that should sent to plugins
            events = {'pupil_positions':pupil_list,'gaze_positions':[]}
            # get world settings
            settings_path = os.path.join(settings_path,'user_settings_world')
            session_world_settings = Persistent_Dict(settings_path)
            default_plugins = [('Dummy_Gaze_Mapper',{})]
            manual_calibration_plugin = [('Manual_Marker_Calibration',{})]
            self.g_pool.plugins = Plugin_List(self.g_pool,plugin_by_name,session_world_settings.get('loaded_plugins',default_plugins)+manual_calibration_plugin)
            self.g_pool.pupil_confidence_threshold = session_world_settings.get('pupil_confidence_threshold',.6)
            self.g_pool.detection_mapping_mode = session_world_settings.get('detection_mapping_mode','2d')

            ''' Step 2: before calculating gaze positions we shall process calibration data
            For calibration we need pupil_list (in events variable) and ref_list - ref_list contains all frames of detected marker
            Using manual_marker_calibration plugin use plugin.update to pass pupil_list and world frames for marker detection
            However, pupil_list is by this point fully detected. Thus, we shall do the following:
            First iteration: send events with all pupil_list with first world frame to manual_marker_calibration plugin.update
            Following iterations: send empty [] pupil_list with next world frame to manual_marker_calibration plugin.update
            '''
            # start calibration - It will always be manual calibration
            cal_plugin = None
            for p in self.g_pool.plugins:
                if 'Manual_Marker_Calibration' in p.class_name:
                    cal_plugin = p
                    break
            cal_plugin.on_notify({'subject':'should_start_calibration'})
            self.active_cal = True

            # read world frames
            w_video_path = [f for f in glob(os.path.join(data_path,"world.*")) if f[-3:] in ('mp4','mkv','avi')][0]
            timestamps_path = os.path.join(data_path, "world_timestamps.npy")
            timestamps = np.load(timestamps_path)

            from ctypes import c_double
            from multiprocessing import Value
            timebase = Value(c_double,0)
            capture_world = autoCreateCapture(w_video_path, timebase=timebase)
            default_settings = {'frame_size':(1280,720),'frame_rate':30}
            capture_world.settings = session_world_settings.get('capture_settings',default_settings)
            # previous_settings = session_world_settings.get('capture_settings',None)
            # if previous_settings and previous_settings['name'] == cap.name:
            #     cap.settings = previous_settings
            # else:
            #     cap.settings = default_settings

            # Test capture
            frame = None
            try:
                frame = capture_world.get_frame()
            except CameraCaptureError:
                logger.error("Could not retrieve image from world.mp4")
                capture_world.close()
                return
            # send first world frame to calibration class via update WITH all pupil_list
            cal_plugin.update(frame,events)

            save_undetected = os.path.join(data_path,'undetected_cal')
            save_detected = os.path.join(data_path,'detected_cal')
            if not os.path.exists(save_detected):
                print "creating %s and save images to"%save_detected
                os.makedirs(save_detected)
            if not os.path.exists(save_undetected):
                print "creating %s and save images to"%save_undetected
                os.makedirs(save_undetected)
            # Send all world frames to calibration class via update WITHOUT pupil_list
            idx = 0
            while cal_plugin.active:
                try:
                    frame = capture_world.get_frame()
                    cal_plugin.update(frame, {'pupil_positions':[],'gaze_positions':[]})
                    detected, pos = self.detect_marker(frame)
                    if pos:
                        self.save_image(frame.img, os.path.join(save_detected,'%s.jpg'%repr(timestamps[idx])), center=pos)
                    else:
                        self.save_image(frame.img, os.path.join(save_undetected,'%s.jpg'%repr(timestamps[idx])))
                    idx += 1
                except EndofVideoFileError:
                    logger.warning("World video file is done. Stopping")
                    break

            ''' Step 3: calculate gaze positions
            passe events to gaze mapper plugin without the world frame
            '''
            for p in self.g_pool.plugins:
                if 'Simple_Gaze_Mapper' in p.class_name:
                    p.update(None,events)
                    break
            save_object(events,os.path.join(data_path, "pupil_data2"))
            timestamps_path = os.path.join(data_path, "world_timestamps.npy")
            timestamps = np.load(timestamps_path)
            pupil_positions_by_frame = None
            pupil_positions_by_frame = correlate_data(pupil_list,timestamps)
        else:
            logger.warning("No eye data found")
            return

        # ++++++++++++

        # pupil_data_path = os.path.join(data_path, "pupil_data")
        # pupil_data = load_object(pupil_data_path)
        #
        #
        #
        #
        #
        #
        # capture_eye = autoCreateCapture(os.path.join(data_path, 'eye0.mp4'), timebase=timebase)
        # default_settings = {'frame_size':(640,480),'frame_rate':60}
        # capture_eye.settings = default_settings
        #
        # # Test capture
        # try:
        #     frame = capture_eye.get_frame()
        # except CameraCaptureError:
        #     logger.error("Could not retrieve image from eye.mp4")
        #     capture_eye.close()
        #     return

        # while True:
        #     # Get an image from the grabber
        #     try:
        #         frame_world = capture_world.get_frame()
        #     except EndofVideoFileError:
        #         logger.warning("World video file is done. Stopping")
        #         break
        #
        # import cv2
        # video_capture = cv2.VideoCapture(os.path.join(data_path, 'world.mp4'))
        # pos_frame = video_capture.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
        # while True:
        #     # get frame by frame
        #     ret, frame = video_capture.read()
        #     # pos_frame = video_capture.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
        #     if video_capture.get(cv2.cv.CV_CAP_PROP_POS_FRAMES) == video_capture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT):
        #         # If the number of captured frames is equal to the total number of frames,
        #         # we stop
        #         break

        # np_arr = np.array([])
        # while True:
        #     # Get an image from the grabber
        #     try:
        #         frame_eye = capture_eye.get_frame()
        #         np_arr = np.append(np_arr, frame_eye.timestamp)
        #     except EndofVideoFileError:
        #         logger.warning("Eye video file is done. Stopping")
        #         break

    def on_notify(self,notification):
        if notification['subject'] is 'rec_stopped':
            self.rec_path = notification['rec_path']
            self.split_recordings()
        elif notification['subject'] is 'cal_stopped':
            self.active_cal = False

    def init_gui(self):
        #lets make a menu entry in the sidebar
        self.menu = ui.Growing_Menu('Split rec2frames')
        self.g_pool.sidebar.append(self.menu)

        #add a button to close the plugin
        self.menu.append(ui.Button('close',self.close))
        # self.menu.append(ui.Text_Input('rec_path',self))
        # # self.menu.append(ui.Text_Input('new_annotation_hotkey',self))
        # self.menu.append(ui.Button('Split rec2frames',self.split_recordings))
        # self.active = True
        # self.sub_menu = ui.Growing_Menu('Events - click to remove')
        # self.menu.append(self.sub_menu)
        # self.update_buttons()


    # def update_buttons(self):
    #     for b in self.buttons:
    #         self.g_pool.quickbar.remove(b)
    #         # self.sub_menu.elements[:] = []
    #     self.buttons = []
    #
    #     for e_name,hotkey in self.annotations:
    #
    #         def make_fire(e_name,hotkey):
    #             return lambda _ : self.fire_annotation(e_name)
    #
    #         def make_remove(e_name,hotkey):
    #             return lambda: self.remove_annotation((e_name,hotkey))
    #
    #         thumb = ui.Thumb(e_name,setter=make_fire(e_name,hotkey), getter=lambda: False,
    #         label=hotkey,hotkey=hotkey)
    #         self.buttons.append(thumb)
    #         self.g_pool.quickbar.append(thumb)
    #         self.sub_menu.append(ui.Button(e_name+" <"+hotkey+">",make_remove(e_name,hotkey)))



    def deinit_gui(self):
        if self.menu:
            self.g_pool.sidebar.remove(self.menu)
            self.menu = None
        if self.buttons:
            for b in self.buttons:
                self.g_pool.quickbar.remove(b)
            self.buttons = []

    def close(self):
        self.alive = False


    # def get_init_dict(self):
    #     return {'annotations':self.annotations}

    def cleanup(self):
        """ called when the plugin gets terminated.
        This happens either voluntarily or forced.
        if you have a GUI or glfw window destroy it here.
        """
        self.deinit_gui()