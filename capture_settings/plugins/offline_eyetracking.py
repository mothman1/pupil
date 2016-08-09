from plugin import Plugin
import logging
import os
import zipfile
import cv2
import numpy as np
from glob import glob
from pyglui import ui
from methods import  normalize, denormalize
from file_methods import Persistent_Dict,load_object, save_object
from video_capture import autoCreateCapture, FileCaptureError, EndofVideoFileError, CameraCaptureError
from PIL import Image, ImageDraw

from circle_detector import find_concetric_circles
import csv
from calibration_routines import calibration_plugins, gaze_mapping_plugins
from plugin import Plugin_List
from calibration_routines import finish_calibration
from ctypes import c_double
from multiprocessing import Value
from requests_futures.sessions import FuturesSession
from uuid import getnode as get_mac
import json
from StringIO import StringIO
from ast import literal_eval as make_tuple
from recorder import Recorder

from enum import Enum
class Workforce(Enum):
    Internally = 0
    OnDemand = 1

logger = logging.getLogger(__name__)

# service_base_uri = 'http://pupilsvc.azurewebsites.net'
service_base_uri = 'http://api.opescode.com/'
test_question_percent= 30
unit_per_job= 100
judgements_unit = 1
payment_cent = 10
max_minutes_job = 10
workforce = Workforce.OnDemand
archive_only = True

def get_api_user():
    mac = get_mac()
    # use hex mac address as the username for this device on the api
    username = format(mac, 'x')
    session = FuturesSession()
    # future = session.get('http://api.opescode.com/api/Users?username=' + username)
    future = session.get('{0}/api/Users?username={1}'.format(service_base_uri, username))
    response = future.result()
    print 'get_api_user response -> ', response.status_code
    logger.info('get_api_user response %s.'%str(response.status_code))
    if response.status_code == 404:
        print 'get_api_user -> user not found. Requesting a new user'
        logger.error('get_api_user -> user not found. Requesting a new user')
        user = {'Username': username, 'Name': username, 'Title': '', 'Email': 'm.othman1@ncl.ac.uk'}
        headers = {'Content-Type': 'application/json'}
        # future = session.post('http://api.opescode.com/api/Users', json=user, headers=headers)
        future = session.post('{0}/api/Users'.format(service_base_uri), json=user, headers=headers)
        response = future.result()
        if response.status_code != 200:
            print 'get_api_user -> user could not be created'
            logger.error('get_api_user -> user could not be created - Request failed with stc=%s'%str(response.status_code))
            return None
        logger.info('get_api_user -> new user (%s) created'%username)
        print 'get_api_user -> created new user %s'%username

    stc = response.status_code
    id = 0
    if stc == 200:
        jsn = response.json()
        id = jsn['Id']
        logger.info('get_api_user -> user id: %s'%str(id))
        print jsn
    else:
        logger.error('get_api_user response -> : %s'%str(stc))
    print 'crowd eye user id: ', id
    return response.json()

def create_api_job(user_id, title, instructions):
    session = FuturesSession()
    member_job = {'UserId': user_id, 'Id': 0, 'Title': title, 'Instructions': instructions}
    headers = {'Content-Type': 'application/json'}
    # future = session.post('http://api.opescode.com/api/MemberJobs', json=member_job, headers=headers)
    future = session.post('{0}/api/MemberJobs'.format(service_base_uri), json=member_job, headers=headers)
    response = future.result()
    stc = response.status_code
    id = 0
    if stc != 200:
        print 'get_api_job -> Job not created - Request failed with stc=%s'%str(stc)
        logger.error('get_api_job -> Job not created - Request failed with stc=%s'%str(stc))
    else:
        jsn = response.json()
        print 'get_api_job -> created job: ', jsn
        id = jsn  # this api method returns integer job id
        logger.info('get_api_job -> Job created - id=%s'%str(id))

    return id
def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for f in files:
            fname, fext = os.path.splitext(f)
            if fext != '.zip':
                ziph.write(os.path.join(root, f))

''' TODO: Add a Question parameter so the questions can be created here rather than on the server. This way we customize
 questions for pupil and world/calibration as needed '''
def crowdsource_undetected(related_list, files_path, instructions, data_for):
    # if no files found then return zero
    if not os.listdir(files_path):
        return 0

    # Remove trailing slashes
    files_path = os.path.normpath(files_path)
    # Get an api crowd user
    api_user = get_api_user()
    crowd_user_id = 0
    if api_user and 'Id' in api_user:
        crowd_user_id = api_user['Id']

    # get a crowd job
    crowd_job_id = 0
    if crowd_user_id > 0:
        crowd_job_id = create_api_job(crowd_user_id, os.path.basename(files_path), instructions)
    zip_path = None
    if crowd_job_id > 0:
        # save json object to json file
        if related_list is not None and len(related_list) > 0:
            sio = StringIO()
            json.dump(related_list, sio)
            with open(os.path.join(files_path,'%s.json'%data_for), "w") as fjson:
                fjson.write(sio.getvalue())
        # compress all files in files_path directory
        zip_path = os.path.join(files_path, '%s.zip'%data_for)
        buff = StringIO()
        with zipfile.ZipFile(buff, 'w', zipfile.ZIP_DEFLATED) as zipf:
            print 'zipping ' + zip_path
            zipdir(files_path, zipf)
            print 'zipped ' + zip_path

        session = FuturesSession()
        # api_uri = 'http://api.opescode.com/api/UserData?id=%s' %str(job_api_id)
        api_uri = '{0}/api/UserData?id={1}'.format(service_base_uri, str(crowd_job_id))
        logger.info('Calling web api {0} for {1}'.format(api_uri, zip_path))

        def bg_cb(sess, resp):
            print zip_path, resp.status_code
            # if failed then save the files to the recording physical folder
            if resp.status_code != 200:
                print 'Post file {0} failed with stc={1}'.format(zip_path, str(resp.status_code))
                # For now, I will not log this until I find a better way to pass logger to the callback method. Note: callback method has no access to self
                logger.error('Post file {0} failed with stc={1}'.format(zip_path, str(resp.status_code)))
            else:
                logger.info('%s posted successfully'%zip_path)
        try:
            with open(zip_path, "wb") as f: # use `wb` mode
                print 'saving zip ' + zip_path
                f.write(buff.getvalue())
                print 'zip saved ' + zip_path
            if not archive_only:
                print 'posting ' + zip_path
                session.post(api_uri, files={"archive": buff.getvalue()}, background_callback=bg_cb)
                print 'posted ' + zip_path
            logger.info('posted %s and awaiting api response.'%zip_path)
        except Exception as ex:
            logger.error('Exception occured while calling web api.')
    return crowd_job_id



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

class offline_eyetracking(Plugin):
    """
    transfer undetected pupil frames to crowdeye api
    """

    def __init__(self,g_pool):
        super(offline_eyetracking, self).__init__(g_pool)
        print 'offline_eyetracking_init'
        self.g_pool = g_pool
        # This is to be the last plugin to initiate - This allows us to disable any pre-loaded plugin as needed
        for p in self.g_pool.plugins:
            if p.class_name != 'Recorder':
                p.alive = False
        self.menu = None
        self.sub_menu = None
        self.buttons = []
        # self.new_annotation_hotkey = 'e'

        self.current_frame = -1
        self.order = 1
        self.active = False
        self.total_frames = 0
        self.rec_path = ''
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

    def save_image_from_array(self, img_array, save_to_file, ellipse=None, center=None, center2=None):
        new_img = Image.fromarray(img_array)
        draw = ImageDraw.Draw(new_img)
        if ellipse is not None:
            center = (int(ellipse['center'][0]),int(ellipse['center'][1]))
            axes = (int(ellipse['axes'][0]/2),int(ellipse['axes'][1]/2))
            draw.ellipse((center[0] - axes[0], center[1] - axes[1], center[0] + axes[0], center[1] + axes[1]), fill = None, outline =255)
        if center and center != (-1,-1):
            diameter = 5
            draw.ellipse((center[0] - diameter, center[1] - diameter, center[0] + diameter, center[1] + diameter), fill=(202,255,90), outline =255)
        if center2 and center2 != (-1,-1):
            diameter = 5
            draw.ellipse((center2[0] - diameter, center2[1] - diameter, center2[0] + diameter, center2[1] + diameter), fill=(90,255,200), outline =255)

        del draw
        new_img.save(save_to_file,quality=20,optimize=True)

    def get_pupil_list(self, crowd_all):
        data_path = self.g_pool.user_dir
        settings_path = os.path.join(self.g_pool.user_settings_path,'user_settings_eye0')
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

        from ui_roi import UIRoi
        self.g_pool.display_mode = session_settings.get('display_mode','camera_image')
        self.g_pool.display_mode_info_text = {'camera_image': "Raw eye camera image. This uses the least amount of CPU power",
                                'roi': "Click and drag on the blue circles to adjust the region of interest. The region should be as small as possible, but large enough to capture all pupil movements.",
                                'algorithm': "Algorithm display mode overlays a visualization of the pupil detection parameters on top of the eye video. Adjust parameters within the Pupil Detection menu below."}
        self.g_pool.u_r = UIRoi(frame.img.shape)
        self.g_pool.u_r.set(session_settings.get('roi',self.g_pool.u_r.get()))
        save_detected = os.path.join(data_path, 'detected_pupil')
        if not os.path.exists(save_detected):
            os.makedirs(save_detected)
        save_undetected = os.path.join(data_path,'undetected_pupil')
        if not os.path.exists(save_undetected):
            os.makedirs(save_undetected)
        pupil_list = []
        # undetected_found = False
        pupil_list_to_crowd = []
        detected_pos1 = None
        detected_pos2 = None
        added_gold_frames = 0
        while True:
            try:
                ''' A first frame was retrived above, we start with processing frames and later get the next one '''
                # pupil ellipse detection
                result = pupil_detector.detect(frame, self.g_pool.u_r, self.g_pool.display_mode == 'algorithm')
                result['id'] = 0
                sequence +=1
                # Use sequence to sort the frames on server and when back from the server
                result['seq'] = sequence
                pupil_list += [result]
                str_timestamp = repr(result['timestamp'])
                crowd_copy = {}
                crowd_copy['Name'] = str_timestamp
                # DataType = 1 is to tell the server this is eye image
                crowd_copy['DataType'] = 1
                crowd_copy['DataFor'] = 8
                crowd_copy['GroupName'] = 'eye'
                crowd_copy['Confidence'] = result['confidence']
                crowd_copy['Additional'] = 'ellipse:%s|diameter:%s'%(str(result['ellipse']),str(result['diameter']))
                pupil_list_to_crowd += [crowd_copy]
                if (not crowd_all and result['confidence'] >0 and result.has_key('ellipse'))\
                        or (result['confidence'] == 1 and added_gold_frames < 50):
                    self.save_image_from_array(frame.img, os.path.join(save_detected,'%s.jpg'%str_timestamp),
                                               result['ellipse'])
                    '''Save 50 high confidence frames as gold images to be used as crowdsourcing quality measure '''
                    if result['confidence'] == 1 and added_gold_frames < 50:
                        self.save_image_from_array(frame.img, os.path.join(save_undetected,'%s.jpg'%str_timestamp))
                        added_gold_frames += 1

                    crowd_copy['Gold'] = True
                    crowd_copy['Demo'] = True
                    if len(pupil_list) > 2 and pupil_list[-2]['confidence'] == 0 and pupil_list[-3]['confidence'] > 0:
                        pupil_list[-2]['norm_pos'] = tuple((p+q)/2 for p, q in zip(pupil_list[-1]['norm_pos'], pupil_list[-3]['norm_pos']))
                        pupil_list_to_crowd[-2]['GoldAnswer'] = str(pupil_list[-2]['norm_pos'])
                else:
                    self.save_image_from_array(frame.img, os.path.join(save_undetected,'%s.jpg'%str_timestamp))
                    # undetected_found = True
                    crowd_copy['Gold'] = False
                    crowd_copy['Demo'] = False
                if result['norm_pos'] != (0.0,1.0):
                    crowd_copy['GoldAnswer'] = str(result['norm_pos'])
                # get frame by frame
                frame = video_capture.get_frame()
            except EndofVideoFileError:
                logger.warning("Eye video file is done. Stopping")
                break
        # keys = pupil_list_to_crowd[0].keys()
        # with open(os.path.join(data_path,'pupil_list.csv'),'wb') as f:
        #     dict_writer = csv.DictWriter(f, keys)
        #     dict_writer.writeheader()
        #     dict_writer.writerows(pupil_list_to_crowd)
        crowd_questions = {}
        crowd_questions['AnswerType'] = 5 #AnswerType.PosXY,
        crowd_questions['GroupName'] = "ET"
        crowd_questions['Instructions'] = "Locate the eye pupil and click in the pupil center (Be as accurate as possible)"
        crowd_questions['AdditionalInfo'] = "If you don't see the pupil please click select 'Not found' button"
        crowd_questions['LabelTxt'] = "Locate the center of eye pupil"
        crowd_questions['QuestionType'] = 8 # QuestionType.ImageTagging
        crowd_questions['Required'] = True
        crowd_questions['Title'] = "Locate pupil"
        ''' Workforce: 0 for internally or 1 or on demand - default is zero (internally) '''
        crowd_data = {'UserDatas': pupil_list_to_crowd, 'Questions': [crowd_questions], 'UnitPerJob': unit_per_job,
                      'JudgementsPerUnit':judgements_unit, 'PaymentCents': payment_cent, 'title': 'Locate the center of eye pupil',
                      'TestQuestPercent': test_question_percent, 'MaxMinutesPerJob': max_minutes_job, 'Workforce': workforce}
        crowd_job_id = crowdsource_undetected(crowd_data, save_undetected, 'Find the center of the eye pupil', 'pupil')
        # Save the job ID to user_info.csv file
        fields=['crowd_pupil_job_id',crowd_job_id]
        with open(os.path.join(data_path, 'user_info.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)

        return pupil_list

    def get_pupil_list_from_csv(self, data_path):

        ''' Detect the pupil after cropping the image based on the crowd located pupil center '''
        data_path = self.g_pool.user_dir
        settings_path = os.path.join(self.g_pool.user_settings_path,'user_settings_eye0')
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

        from ui_roi import UIRoi
        self.g_pool.display_mode = session_settings.get('display_mode','camera_image')
        self.g_pool.display_mode_info_text = {'camera_image': "Raw eye camera image. This uses the least amount of CPU power",
                                'roi': "Click and drag on the blue circles to adjust the region of interest. The region should be as small as possible, but large enough to capture all pupil movements.",
                                'algorithm': "Algorithm display mode overlays a visualization of the pupil detection parameters on top of the eye video. Adjust parameters within the Pupil Detection menu below."}

        # Get 75px before and after the crowd located pupil center -> the cropped image is 150X150 pixels
        # crop_width = 300
        # crop_height = 225
        self.g_pool.u_r = UIRoi((480,640,3))
        # self.g_pool.u_r.set(session_settings.get('roi',self.g_pool.u_r.get()))
        ''' End '''

        pupil_list = []
        with open(os.path.join(data_path, 'crowdpos/eye.csv'), 'rU') as csvfile:
            all = csv.reader(csvfile, delimiter=',')
            for row in all:
                norm_center = make_tuple(row[1])
                center = (norm_center[0] * 640, norm_center[1] * 480)
                prow = {'timestamp':float(row[0]), 'confidence':1, 'center':(center[0],center[1]), 'norm_pos': (norm_center[0],norm_center[1]), 'id':0, 'method': '2d c++'}
                pupil_list.append(prow)

        timebase = Value(c_double,0)
        capture_eye = autoCreateCapture(os.path.join(data_path, 'eye0.mp4'), timebase=timebase)
        default_settings = {'frame_size':(640,480),'frame_rate':30}
        capture_eye.settings = default_settings
        # import cv2
        # video_capture = cv2.VideoCapture('/Developments/NCLUni/pupil_crowd4Jul16/recordings/2016_07_06/003/world.mp4')
        # pos_frame = video_capture.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
        save_crowd_detected = os.path.join(data_path, 'crowd_detected_pupil')
        if not os.path.exists(save_crowd_detected):
            os.makedirs(save_crowd_detected)
        idx = 0
        count_detected = 0
        pupil_list_ret = []
        while len(pupil_list) > idx:
            try:
                # get frame by frame
                frame = capture_eye.get_frame()
                # pos_frame = video_capture.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
                r_idx, related_csv_line = self.search(frame.timestamp, pupil_list)
                if related_csv_line:
                    norm_pos = related_csv_line["norm_pos"]
                    center = (norm_pos[0] * 640, (1 - norm_pos[1]) * 480)
                    crowd_center = (center[0], center[1])

                    ''' auto detect the center and ellipse after cropping the image based on the crowd located pupil '''
                    lx,ly,ux,uy = center[0] - 75,center[1]-75,center[0]+75,center[1]+75
                    ellipse = None
                    if lx >= 1 and ly >= 1 and ux < 640 and uy < 480:
                        self.g_pool.u_r.set((lx,ly,ux,uy, (480,640,3)))
                        # orig_img = frame.img
                        # orig_gray = frame.gray
                        # print frame.img.flags
                        # print frame.img.shape
                        # print '---'
                        # print frame.gray.flags
                        # print frame.gray.shape
                        # print '---'
                        # cropped_img_nparr = frame.img[ly:uy,lx:ux].copy(order='C')
                        # frame._img = cropped_img_nparr
                        # cropped_img = Image.fromarray(cropped_img_nparr)
                        # frame._gray = frame.gray[ly:uy,lx:ux].copy(order='C')
                        # print frame.img.flags
                        # print frame.img.shape
                        # print '---'
                        # print frame.gray.flags
                        # print frame.gray.shape
                        # frame.width = crop_width
                        # frame.height = crop_height
                        # print frame.width, 'x', frame.height
                        result = pupil_detector.detect(frame, self.g_pool.u_r, self.g_pool.display_mode == 'algorithm')
                        result['id'] = 0
                        # cropped_center = (result["norm_pos"][0]*crop_width,result["norm_pos"][0]*crop_height)
                        # crowd_auto_center = (result["norm_pos"][0]*640,result["norm_pos"][0]*480)
                        # self.save_image_from_array(frame.img, os.path.join(save_crowd_detected, '%s_cropped.jpg'%repr(frame.timestamp)), center=cropped_center)
                        # self.save_image_from_array(orig_img, os.path.join(save_crowd_detected, '%s_crowd_auto.jpg'%repr(frame.timestamp)), center=crowd_auto_center)
                        if result['ellipse'] is not None and result['confidence'] > 0:
                            ellipse =result['ellipse']
                            center = (ellipse['center'][0],ellipse['center'][1])
                            norm_pos = (result['norm_pos'][0], result['norm_pos'][1])
                            count_detected += 1
                            related_csv_line['ellipse'] = ellipse
                        ''' End auto detect '''

                    related_csv_line['center'] = (center[0], center[1])
                    related_csv_line['norm_pos'] = (norm_pos[0], norm_pos[1])
                    pupil_list_ret.append(related_csv_line)
                    # pupil_list[r_idx]['center'] = center
                    self.save_image_from_array(frame.img, os.path.join(save_crowd_detected, '%s.jpg'%repr(frame.timestamp)), center=center, ellipse=ellipse, center2=crowd_center)
                    idx += 1
            except EndofVideoFileError:
                logger.warning("World video file is done. Stopping")
                break
            except Exception as ex:
                exc = ex.message

        print 'Dtected pupils from crowd initial input = ', count_detected
        # pupil_list.sort(key=lambda d: d['timestamp'])
        # return pupil_list
        return pupil_list_ret

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

    def calibrate(self,events, data_path, session_world_settings, crowd_all):
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
        # timestamps_path = os.path.join(data_path, "world_timestamps.npy")
        # timestamps = np.l os.path.join(data_path, "world_timestamps.npy")
        # timestamps = np.load(timestamps_path)

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
        added_idx = 0
        try:
            frame = capture_world.get_frame()
        except CameraCaptureError:
            logger.error("Could not retrieve image from world.mp4")
            capture_world.close()
            return
        # send first world frame to calibration class via update WITH all pupil_list in events - This will update
        # all pupil_list in the calibration plugin
        save_undetected = os.path.join(data_path,'undetected_cal')
        save_detected = os.path.join(data_path,'detected_cal')
        if not os.path.exists(save_detected):
            print "creating %s and save images to"%save_detected
            os.makedirs(save_detected)
        if not os.path.exists(save_undetected):
            print "creating %s and save images to"%save_undetected
            os.makedirs(save_undetected)
        ref_list = []
        cal_plugin.update(frame,events)
        # if len(cal_plugin.ref_list) > added_idx:
        #     ref = cal_plugin.ref_list[added_idx - 1]
        #     ref_list.append(ref)
        #     self.save_image_from_array(frame.img, os.path.join(save_detected,'%s.jpg'%repr(timestamps[frame_idx])), center=ref['screen_pos'])
        #     added_idx += 1
        # else:
        #     self.save_image_from_array(frame.img, os.path.join(save_undetected,'%s.jpg'%repr(timestamps[frame_idx])))
        # Send all world frames to calibration class via update WITHOUT pupil_list as we already sent pupil_list in
        # the first call to calibrate above
        ref_list_to_crowd = []
        save_to = save_detected
        added_gold_frames = 0
        while cal_plugin.active:
            try:
                gold = True
                Demo = True
                ref = {}
                # detected, pos = self.detect_marker(frame)
                if len(cal_plugin.ref_list) > added_idx:
                    ref = cal_plugin.ref_list[added_idx - 1]
                    added_idx += 1
                    ref_list.append(ref)
                    save_to = save_detected if not crowd_all else save_undetected
                else:
                    ref["norm_pos"] = (-1,-1)
                    ref["screen_pos"] = (-1,-1)
                    ref["timestamp"] = frame.timestamp
                    save_to = save_undetected
                    gold = False
                    demo = False

                crowd_copy = {}
                str_timestamp = repr(frame.timestamp)
                crowd_copy['Name'] = str_timestamp
                # DataType = 6 tells the server this is world image
                crowd_copy['DataType'] = 6
                crowd_copy['DataFor'] = 8
                crowd_copy['GoldAnswer'] = str(ref['norm_pos'])
                crowd_copy['GroupName'] = 'Cal'
                crowd_copy['Gold'] = gold if not crowd_all else False
                crowd_copy['Demo'] = demo if not crowd_all else False
                '''Save 50 high confidence frames as gold images to be used as crowdsourcing quality measure '''
                if gold and added_gold_frames <= 50:
                    self.save_image_from_array(frame.img, os.path.join(save_undetected,'%s.jpg'%str_timestamp))
                    self.save_image_from_array(frame.img, os.path.join(save_detected,'%s.jpg'%str_timestamp), center=ref['screen_pos'])
                    added_gold_frames += 1
                    crowd_copy['Gold'] = gold
                    crowd_copy['Demo'] = demo

                ref_list_to_crowd += [crowd_copy]
                self.save_image_from_array(frame.img, os.path.join(save_to,'%s.jpg'%str_timestamp), center=ref['screen_pos'] if not crowd_all else None)
                frame = capture_world.get_frame()
                cal_plugin.update(frame, {'pupil_positions':[],'gaze_positions':[]})
            except EndofVideoFileError:
                logger.warning("World video file is done. Stopping")
                break

        # keys = ref_list_to_crowd[0].keys()
        # with open(os.path.join(data_path,'ref_list.csv'),'wb') as f:
        #     dict_writer = csv.DictWriter(f, keys)
        #     dict_writer.writeheader()
        #     dict_writer.writerows(ref_list_to_crowd)
        crowd_questions = {}
        crowd_questions['AnswerType'] = 5 #AnswerType.PosXY,
        crowd_questions['GroupName'] = "ET"
        crowd_questions['Instructions'] = "Locate the center of a nested three-black-circle marker and click in the pupil center (Be as accurate as possible)"
        crowd_questions['AdditionalInfo'] = "If you don't see the nested three-black-circle please select 'No marker' tick box"
        crowd_questions['LabelTxt'] = "Locate the nested three-black-circle marker"
        crowd_questions['QuestionType'] = 8 # QuestionType.ImageTagging,
        crowd_questions['Required'] = True
        crowd_questions['Title'] = "Locate marker"
        ''' Workforce: 0 for internally or 1 or on demand - default is zero (internally) '''
        crowd_job = {'UserDatas': ref_list_to_crowd, 'Questions': [crowd_questions], 'UnitPerJob': unit_per_job,
                     'JudgementsPerUnit':judgements_unit, 'PaymentCents': payment_cent, 'title': 'Locate the center of the circular marker',
                     'TestQuestPercent': test_question_percent,'MaxMinutesPerJob': max_minutes_job, 'Workforce': workforce}
        crowd_job_id = crowdsource_undetected(crowd_job, save_undetected, 'Find the center of the nested three-black-circle marker', 'cal')
        # Save the job ID to user_info.csv file
        fields=['crowd_cal_job_id',crowd_job_id]
        with open(os.path.join(data_path, 'user_info.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)

    def calibrate_from_csv(self, pupil_list, data_path):
        import csv
        ref_list = []
        all_ref_list = []
        with open(os.path.join(data_path, 'crowdpos/cal.csv'), 'rU') as csvfile:
            all = csv.reader(csvfile, delimiter=',')
            smooth_pos1 = 0.,0.
            smooth_vel1 = 0
            sample_site1 = (-2,-2)
            counter1 = 0
            counter_max = 30
            count_all_detected = 0
            for row in all:
                norm_center = make_tuple(row[1])
                center = (norm_center[0] * 1280, 1 - norm_center[1] * 720)
                center = (int(round(center[0])),int(round(center[1])))

                # calculate smoothed manhattan velocity
                smoother = 0.3
                smooth_pos = np.array(smooth_pos1)
                pos = np.array(norm_center)
                new_smooth_pos = smooth_pos + smoother*(pos-smooth_pos)
                smooth_vel_vec = new_smooth_pos - smooth_pos
                smooth_pos = new_smooth_pos
                smooth_pos1 = list(smooth_pos)
                #manhattan distance for velocity
                new_vel = abs(smooth_vel_vec[0])+abs(smooth_vel_vec[1])
                smooth_vel1 = smooth_vel1 + smoother*(new_vel-smooth_vel1)

                #distance to last sampled site
                sample_ref_dist = smooth_pos-np.array(sample_site1)
                sample_ref_dist = abs(sample_ref_dist[0])+abs(sample_ref_dist[1])

                # start counter if ref is resting in place and not at last sample site
                if not counter1:

                    if smooth_vel1 < 0.01 and sample_ref_dist > 0.1:
                        sample_site1 = smooth_pos1
                        logger.debug("Steady marker found. Starting to sample %s datapoints" %counter_max)
                        # self.notify_all({'subject':'calibration marker found','timestamp':self.g_pool.capture.get_timestamp(),'record':True,'network_propagate':True})
                        counter1 = counter_max

                if counter1:
                    if smooth_vel1 > 0.01:
                        logger.warning("Marker moved too quickly: Aborted sample. Sampled %s datapoints. Looking for steady marker again."%(counter_max-counter1))
                        # self.notify_all({'subject':'calibration marker moved too quickly','timestamp':self.g_pool.capture.get_timestamp(),'record':True,'network_propagate':True})
                        counter1 = 0
                    else:
                        count_all_detected += 1
                        counter1 -= 1
                        ref = {}
                        ref["norm_pos"] = norm_center
                        ref["screen_pos"] = center
                        ref["timestamp"] = float(row[0])
                        ref_list.append(ref)
                        if counter1 == 0:
                            #last sample before counter done and moving on
                            logger.debug("Sampled %s datapoints. Stopping to sample. Looking for steady marker again."%counter_max)
                            # self.notify_all({'subject':'calibration marker sample completed','timestamp':self.g_pool.capture.get_timestamp(),'record':True,'network_propagate':True})
                # save all ref to look at pos on the images
                ref = {}
                ref["norm_pos"] = norm_center
                ref["screen_pos"] = center
                ref["timestamp"] = float(row[0])
                all_ref_list.append(ref)

        ref_list.sort(key=lambda d: d['timestamp'])
        all_ref_list.sort(key=lambda d: d['timestamp'])
        timebase = Value(c_double,0)
        capture_world = autoCreateCapture(os.path.join(data_path, 'world.mp4'), timebase=timebase)
        default_settings = {'frame_size':(1280,720),'frame_rate':30}
        capture_world.settings = default_settings
        # import cv2
        # video_capture = cv2.VideoCapture('/Developments/NCLUni/pupil_crowd4Jul16/recordings/2016_07_06/003/world.mp4')
        # pos_frame = video_capture.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
        save_crowd_detected = os.path.join(data_path, 'crowd_detected_cal')
        if not os.path.exists(save_crowd_detected):
            os.makedirs(save_crowd_detected)
        while True:
            try:
                # get frame by frame
                frame = capture_world.get_frame()
                r_idx, selected_frame = self.search(frame.timestamp, all_ref_list )
                if selected_frame:
                    # pos_frame = video_capture.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
                    center = (selected_frame["norm_pos"][0] * 1280, (1 - selected_frame["norm_pos"][1]) * 720)
                    self.save_image_from_array(frame.img, save_crowd_detected + '/%s.jpg'%repr(frame.timestamp), center=center)
            except EndofVideoFileError:
                logger.warning("World video file is done. Stopping")
                break
            except:
                break
        keys = all_ref_list[0].keys()
        with open(os.path.join(data_path, 'crowdpos/generated_ref_list.csv'),'wb') as f:
            dict_writer = csv.DictWriter(f, keys)
            dict_writer.writeheader()
            dict_writer.writerows(all_ref_list)
        finish_calibration.finish_calibration(self.g_pool,pupil_list,ref_list)
        return ref_list

    def calibrate_from_user_calibration_data_file(self):
        user_calibration = load_object(os.path.join(self.g_pool.user_dir, "user_calibration_data"))

        self.pupil_list = user_calibration['pupil_list']
        self.ref_list = user_calibration['ref_list']
        calibration_method = user_calibration['calibration_method']

        if '3d' in calibration_method:
            logger.error('adjust calibration is not supported for 3d calibration.')
            return


        finish_calibration.finish_calibration(self.g_pool,self.pupil_list,self.ref_list)

    def search(self, ts, dict_lst):
        idx = 0
        for i in dict_lst:
            if i['timestamp'] == ts:
                return idx, i
            idx += 1
        return -1, None

    def start_processing(self):
        data_path = '/Developments/NCLUni/pupil_crowd4Jul16/recordings/2016_07_29/003Crowd2CircleCursor'#self.rec_path
        # Set user_dir to data_path so all related plugins save to the same folder as the recordings
        self.g_pool.user_dir = data_path
        # Manage plugins
        plugin_by_index = [Recorder]+calibration_plugins+gaze_mapping_plugins
        name_by_index = [p.__name__ for p in plugin_by_index]
        plugin_by_name = dict(zip(name_by_index,plugin_by_index))

        self.g_pool.user_settings_path = os.path.join(data_path[:data_path.index('recordings')], 'capture_settings')

        ''' Step 1: when possible detect all pupil positions '''
        pupil_list = self.get_pupil_list(crowd_all=True)
        # pupil_list = self.get_pupil_list_from_csv(data_path)
        # pupil_list = []
        if pupil_list:
            # create events variable that should sent to plugins
            events = {'pupil_positions':pupil_list,'gaze_positions':[]}
            # get world settings
            user_settings_world_path = os.path.join(self.g_pool.user_settings_path,'user_settings_world')
            user_settings_world = Persistent_Dict(user_settings_world_path)
            default_plugins = [('Recorder',{})]
            simple_gaze_mapper = [('Simple_Gaze_Mapper',{})]
            manual_calibration_plugin = [('Manual_Marker_Calibration',{})]
            self.g_pool.plugins = Plugin_List(self.g_pool,plugin_by_name,user_settings_world.get('loaded_plugins',default_plugins)+manual_calibration_plugin)
            # self.g_pool.plugins.add(simple_gaze_mapper)
            self.g_pool.pupil_confidence_threshold = user_settings_world.get('pupil_confidence_threshold',.6)
            self.g_pool.detection_mapping_mode = user_settings_world.get('detection_mapping_mode','2d')

            ''' Step 2: before calculating gaze positions we shall process calibration data
            For calibration we need pupil_list (in events variable) and ref_list - ref_list contains all frames of detected marker
            Using manual_marker_calibration plugin use plugin.update to pass pupil_list and world frames for marker detection
            However, pupil_list is by this point fully detected. Thus, we shall do the following:
            First iteration: send events with all pupil_list with first world frame to manual_marker_calibration plugin.update
            Following iterations: send empty [] pupil_list with next world frame to manual_marker_calibration plugin.update
            '''
            self.calibrate(events, data_path, user_settings_world, crowd_all=True)
            # self.calibrate_from_csv(pupil_list, data_path)
            # self.calibrate_from_user_calibration_data_file()

            ''' Step 3: calculate gaze positions
            passe events to gaze mapper plugin without the world frame
            '''
            for p in self.g_pool.plugins:
                if 'Simple_Gaze_Mapper' in p.class_name:
                    p.update(None,events)
                    break
            save_object(events,os.path.join(data_path, "pupil_data"))
            # timestamps_path = os.path.join(data_path, "world_timestamps.npy")
            # timestamps = np.load(timestamps_path)
            # pupil_positions_by_frame = None
            # pupil_positions_by_frame = correlate_data(pupil_list,timestamps)
        else:
            logger.warning("No eye data found")

        # Again, remove all plugins except recorder
        for p in self.g_pool.plugins:
            if p.class_name != 'Recorder':
                p.alive = False

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

    def on_notify(self,notification):
        if notification['subject'] is 'rec_stopped':
            self.rec_path = notification['rec_path']
            self.start_processing()
        elif notification['subject'] is 'cal_stopped':
            self.active_cal = False

    def init_gui(self):
        #lets make a menu entry in the sidebar
        self.menu = ui.Growing_Menu('Offline eye tracking')
        self.g_pool.sidebar.append(self.menu)

        #add a button to close the plugin
        self.menu.append(ui.Button('close',self.close))
        self.menu.append(ui.Text_Input('rec_path',self))
        # self.menu.append(ui.Text_Input('new_annotation_hotkey',self))
        self.menu.append(ui.Button('Process',self.start_processing))
        self.active = True
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
        # if self.buttons:
        #     for b in self.buttons:
        #         self.g_pool.quickbar.remove(b)
        #     self.buttons = []

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