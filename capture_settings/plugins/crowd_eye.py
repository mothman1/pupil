from plugin import Plugin
import StringIO
import logging
from pyglui import ui
from PIL import Image
import os
import zipfile
from requests_futures.sessions import FuturesSession
from uuid import getnode as get_mac
import json
logger = logging.getLogger(__name__)

# service_base_uri = 'http://api.opescode.com'
service_base_uri = 'http://api.opescode.com'

class InMemoryZip(object):
    def __init__(self):
        # Create the in-memory file-like object
        self.in_memory_data = StringIO.StringIO()
        # Get a handle to the in-memory zip in append mode
        self.in_memory_zip = zipfile.ZipFile(self.in_memory_data, "a", zipfile.ZIP_DEFLATED, False)

    def append(self, filename_in_zip, file_contents):
        '''Appends a file with name filename_in_zip and contents of
        file_contents to the in-memory zip.'''

        # Write the file to the in-memory zip
        img = Image.fromarray(file_contents)
        image_file = StringIO.StringIO()
        img.save(image_file, 'JPEG', quality=50)
        self.in_memory_zip.writestr(filename_in_zip, image_file.getvalue())
        return self

    def read(self):
        '''Returns a string with the contents of the in-memory zip.'''
        self.in_memory_data.seek(0)
        return self.in_memory_data.read()

    def post_to_crowd_eye_api(self, filename, job_api_id, pupil_pos = None):
        if pupil_pos is not None and len(pupil_pos) > 0:
            sio = StringIO.StringIO()
            json.dump(pupil_pos, sio)
            self.in_memory_zip.writestr("pupil_pos.json", sio.getvalue())
        # Mark the files as having been created on Windows so that
        # Unix permissions are not inferred as 0000
        for zfile in self.in_memory_zip.filelist:
            zfile.create_system = 0
        '''Writes the in-memory zip to a file.'''
        # zipfile must be closed first
        self.in_memory_zip.close()
        # post request to api
        session = FuturesSession()
        # api_uri = 'http://api.opescode.com/api/UserData?id=%s' %str(job_api_id)
        api_uri = '{0}/api/UserData?id={1}'.format(service_base_uri, str(job_api_id))
        logger.info('Calling web api {0}. for file {1}'.format(api_uri, filename))
        def bg_cb(sess, resp):
            print filename, resp.status_code
            # if failed then save the files to the recording physical folder
            if resp.status_code != 200:
                print 'Post file {0} failed with stc={1}'.format(filename, str(resp.status_code))
                # For now, I will not log this until I find a better way to pass logger to the callback method. Note: callback method has no access to self
                logger.error('Post file {0} failed with stc={1}'.format(filename, str(resp.status_code)))
                with open(filename, 'wb') as f:
                    f.write(zipData)
            else:
                logger.info('%s posted successfully'%filename)
        try:
            session.post(api_uri, files={"archive": ("pupil.zip", self.read())}, background_callback=bg_cb)
            zipData = self.read()
            logger.info('posted %s but awaiting api response.'%filename)
        except Exception as ex:
            logger.error('Exception occured while calling web api.')
        # with open(filename, 'wb') as f:
        #     f.write(self.in_memory_data.getvalue())
        self.in_memory_data.close()

class crowd_eye(Plugin):
    """
    transfer undetected pupil frames to crowdeye api
    """

    def __init__(self, g_pool, crowdeye_active=True):
        super(crowd_eye, self).__init__(g_pool)
        print 'crowd_eye_init'
        self.order = .9
        self.imz = InMemoryZip()
        self.crowdeye_active = crowdeye_active
        # self.recorder_plugin = [r for r in self.g_pool.plugins if r.class_name == 'Recorder'][0]
        # self.calibration_plugin = [c for c in self.g_pool.plugins if '_calibration' in c.class_name][0]
        self.total_frames = 0
        self.recording = False
        self.calibrating = False
        self.undetected_count = 0
        self.detected_pupil_pos = []
        self.save_detected_pupil_frame = True
        self.detected_count = 0
        self.rec_path = ''
        json_resp = self.GetApiUser()
        self.crowd_eye_user_id = 0
        if 'Id' in json_resp:
            self.crowd_eye_user_id = json_resp['Id']
        self.crowd_eye_job_id = 0
        self.created_jobs_count = 0

    def GetApiUser(self):
        mac = get_mac()
        # use hex mac address as the username for this device on the api
        username = format(mac, 'x')
        session = FuturesSession()
        # future = session.get('http://api.opescode.com/api/Users?username=' + username)
        future = session.get('{0}/api/Users?username={1}'.format(service_base_uri, username))
        response = future.result()
        print 'GetApiUser response -> ', response.status_code
        logger.info('GetApiUser response %s.'%str(response.status_code))
        if response.status_code == 404:
            print 'GetApiUser -> user not found'
            logger.error('GetApiUser -> user not found. Requesting a new user')
            user = {'Username': username, 'Name': username, 'Title': '', 'Email': 'm.othman1@ncl.ac.uk'}
            headers = {'Content-Type': 'application/json'}
            # future = session.post('http://api.opescode.com/api/Users', json=user, headers=headers)
            future = session.post('{0}/api/Users'.format(service_base_uri), json=user, headers=headers)
            response = future.result()
            if response.status_code != 200:
                print 'GetApiUser -> user could not be created'
                logger.error('GetApiUser -> user could not be created - Request failed with stc=%s'%str(response.status_code))
            logger.info('GetApiUser -> new user created')
            print 'GetApiUser -> created new user ->', response.status_code

        stc = response.status_code
        id = 0
        if stc == 200:
            jsn = response.json()
            id = jsn['Id']
            logger.info('GetApiUser -> user id: %s'%str(id))
            print jsn
        else:
            logger.error('GetApiUser response -> : %s'%str(stc))
        print 'crowd eye user id: ', id
        return response.json()

    def CreateApiJob(self, user_id, title):
        session = FuturesSession()
        member_job = {'UserId': user_id, 'Id': 0, 'Title': title, 'Instructions': 'Find the center of the eye pupil'}
        headers = {'Content-Type': 'application/json'}
        # future = session.post('http://api.opescode.com/api/MemberJobs', json=member_job, headers=headers)
        future = session.post('{0}/api/MemberJobs'.format(service_base_uri), json=member_job, headers=headers)
        response = future.result()
        stc = response.status_code
        print 'CreateApiJob response -> ', stc
        id = 0
        if stc != 200:
            logger.error('CreateApiJob -> Job not created - Request failed with stc=%s'%str(stc))
        else:
            jsn = response.json()
            print 'CreateApiJob -> created job: ', jsn
            id = jsn  # this api method returns integer job id
            logger.info('CreateApiJob -> Job created - id=%s'%str(id))

        return id


    def on_notify(self,notification):
        if notification['subject'] is 'rec_started':
            # TODO: create or get user from crowd eye api
            self.calibrating = False
            self.undetected_count = 0
            self.detected_pupil_pos = []
            self.save_detected_pupil_frame = True
            self.detected_count = 0
            if self.crowd_eye_user_id == 0:
                json_resp = self.GetApiUser()
                if 'Id' in json_resp:
                    self.crowd_eye_user_id = json_resp['Id']
            self.created_jobs_count = 0
            self.recording = True
            self.rec_path = notification['rec_path']
            dirs = os.path.split(os.path.abspath(notification['rec_path']))
            self.crowd_eye_job_id = self.CreateApiJob(self.crowd_eye_user_id, '{0}_{1}'.format(os.path.split(os.path.abspath(dirs[0]))[1], dirs[1]))
            # start calibration when recording starts
            self.notify_all({'subject':'should_start_calibration'})
            # import zmq
            # from time import sleep,time
            # with zmq.Context() as context:
            #     with context.socket(zmq.REQ) as socket:
            #         # set your ip here
            #         socket.connect('tcp://127.0.0.1:50020')
            #         t= time()
            #         socket.send('C')
            #         print socket.recv()
            #         print 'Round trip command delay:', time()-t
        elif notification['subject'] is 'rec_stopped':
            # TODO: Finish the job or set a loop with timer to check on the api jobs completion
            if self.undetected_count >= 0 or self.detected_count >= 0:
                ''' We dont need to save all detected frames - For now, I save just 100 detected frames. For the rest we
                create a json file (smaller to send over the internet) '''
                self.imz.post_to_crowd_eye_api(os.path.join(self.rec_path, "undetected%s.zip"%str(self.created_jobs_count)), self.crowd_eye_job_id, self.detected_pupil_pos)
                # reset API post data
                self.imz = InMemoryZip()
            self.recording = False
            self.crowd_eye_job_id = 0
            print 'total frames_crowd_eye', self.total_frames
        # Remote has stopped recording, we should stop as well.
        if notification['subject'] is 'should_start_calibration':
            self.calibrating = True
        elif notification['subject'] in ('should_stop_calibration', 'cal_stopped'):
            self.calibrating = False


    '''frame: the world frame
    events: contains pupil_position frames and data, as well as gaze positions and dt'''
    def update(self,frame,events):
        # print 'crowd_eye_update'
        is_rec = self.recording
        is_cal = self.calibrating
        if is_rec or is_cal:
            cal_rec = 'cal' if is_cal else 'rec'
            recent_pupil_pos = events['pupil_positions']
            ''' Accept all pupil frames from calibration whether marker is detected or not
                There shouldnt be too many frames to crowdsource from calibration
                and we will anyway dismiss the non-correlated ones with the detected markers timestamp '''
            #  always save pupil positions
            for p_pt in recent_pupil_pos:
                self.total_frames += 1
                if p_pt["confidence"] == 0.0:
                    pos_xy = "{0}X{1}".format(p_pt["ellipse"]["center"][0], p_pt["ellipse"]["center"][1])
                    self.imz.append('{0}_{1}_{2}_{3}_{4}_eye.jpg'.format(repr(p_pt["timestamp"]), p_pt["confidence"], pos_xy, 0, cal_rec), p_pt['eye_img'])
                    self.undetected_count += 1
                # We dont need all details for the detected frame - especially we dont need the image so we dont overload the memory
                else:
                    if self.save_detected_pupil_frame and self.detected_count <= 100:
                        pos_xy = "{0}X{1}".format(p_pt["ellipse"]["center"][0], p_pt["ellipse"]["center"][1])
                        self.imz.append('{0}_{1}_{2}_{3}_{4}_eye.jpg'.format(repr(p_pt["timestamp"]), p_pt["confidence"], pos_xy, 1, cal_rec), p_pt['eye_img'])
                    else:
                        p = {"Confidence": p_pt["confidence"], "Timestamp": repr(p_pt["timestamp"]),
                                 "PosXY": "{0}X{1}".format(p_pt["ellipse"]["center"][0], p_pt["ellipse"]["center"][1]),
                                 "GroupName": cal_rec, "DataFor": 8, "DataType": 1, "Gold": True, "Demo": True}
                        self.detected_pupil_pos.append(p)
                    self.detected_count += 1
                p_pt['eye_img'] = []
                # End loop
            if self.undetected_count >= 100 and self.detected_count >= 100:
                ''' We dont need to save all detected frames - For now, I save just 100 detected frames. For the rest we
                create a json file (smaller to send over the internet) '''
                self.save_detected_pupil_frame = False
                self.detected_count = 0
                self.undetected_count = 0
                self.imz.post_to_crowd_eye_api(os.path.join(self.rec_path, "undetected%s.zip"%str(self.created_jobs_count)), self.crowd_eye_job_id, self.detected_pupil_pos)
                self.detected_pupil_pos = []
                self.created_jobs_count += 1
                # reset API post data
                self.imz = InMemoryZip()

    def init_gui(self):
        self.crowdeye_switch = ui.Switch('crowdeye_active',self,label='Crowdsource undetected pupil frames')
        self.g_pool.sidebar[0].insert(-1,self.crowdeye_switch)

    def deinit_gui(self):
        if self.crowdeye_switch:
            self.g_pool.sidebar[0].remove(self.crowdeye_switch)
            self.crowdeye_switch = None

    def cleanup(self):
        self.deinit_gui()

    # def gl_display(self):
    #     for pt,a in self.pupil_display_list:
    #         #This could be faster if there would be a method to also add multiple colors per point
    #         draw_points_norm([pt],
    #                     size=35,
    #                     color=RGBA(1.,.2,.4,a))

    def get_init_dict(self):
        return {'crowdeye_active':True}