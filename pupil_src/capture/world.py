'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''
import os, sys, platform


class Global_Container(object):
    pass

def world(pupil_queue,timebase,launcher_pipe,eye_pipes,eyes_are_alive,user_dir,version,cap_src):
    """world
    Creates a window, gl context.
    Grabs images from a capture.
    Receives Pupil coordinates from eye process[es]
    Can run various plug-ins.
    """

    import logging
    # Set up root logger for this process before doing imports of logged modules.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    #silence noisy modules
    logging.getLogger("OpenGL").setLevel(logging.ERROR)
    # create formatter
    formatter = logging.Formatter('%(processName)s - [%(levelname)s] %(name)s : %(message)s')
    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(user_dir,'capture.log'),mode='w')
    fh.setLevel(logger.level)
    fh.setFormatter(formatter)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logger.level+10)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)


    #setup thread to recv log recrods from other processes.
    def log_loop(logging):
        import zmq
        ctx = zmq.Context()
        sub = ctx.socket(zmq.SUB)
        sub.bind('tcp://127.0.0.1:502020')
        sub.setsockopt(zmq.SUBSCRIBE, "")
        while True:
            record = sub.recv_pyobj()
            logger = logging.getLogger(record.name)
            logger.handle(record)

    import threading
    log_thread = threading.Thread(target=log_loop, args=(logging,))
    log_thread.setDaemon(True)
    log_thread.start()


    # create logger for the context of this function
    logger = logging.getLogger(__name__)


    # We defer the imports because of multiprocessing.
    # Otherwise the world process each process also loads the other imports.
    # This is not harmful but unnecessary.

    #general imports
    from time import time,sleep
    import numpy as np

    from uvc import get_time_monotonic

    def get_timestamp():
        return get_time_monotonic()-timebase.value

    # helpers/utils
    from file_methods import Persistent_Dict
    from methods import delta_t, get_system_info
    from version_utils import VersionFormat

    # Plug-ins
    from plugin import Plugin_List,import_runtime_plugins
    from calibration_routines import calibration_plugins, gaze_mapping_plugins
    from pupil_server import Pupil_Server
    from pupil_sync import Pupil_Sync
    from pupil_remote import Pupil_Remote

    logger.info('Application Version: %s'%version)
    logger.info('System Info: %s'%get_system_info())


    #g_pool holds variables for this process
    g_pool = Global_Container()

    # make some constants avaiable
    g_pool.user_dir = user_dir
    g_pool.version = version
    g_pool.app = 'service'
    g_pool.pupil_queue = pupil_queue
    g_pool.timebase = timebase
    # g_pool.launcher_pipe = launcher_pipe
    g_pool.eye_pipes = eye_pipes
    g_pool.eyes_are_alive = eyes_are_alive


    #manage plugins
    runtime_plugins = import_runtime_plugins(os.path.join(g_pool.user_dir,'plugins'))
    user_launchable_plugins = []+runtime_plugins
    system_plugins  = []
    plugin_by_index =  system_plugins+user_launchable_plugins+calibration_plugins+gaze_mapping_plugins
    name_by_index = [p.__name__ for p in plugin_by_index]
    plugin_by_name = dict(zip(name_by_index,plugin_by_index))
    default_plugins = []




    tick = delta_t()
    def get_dt():
        return next(tick)

    # load session persistent settings
    session_settings = Persistent_Dict(os.path.join(g_pool.user_dir,'user_settings_world'))
    if session_settings.get("version",VersionFormat('0.0')) < g_pool.version:
        logger.info("Session setting are from older version of this app. I will not use those.")
        session_settings.clear()


    g_pool.pupil_confidence_threshold = session_settings.get('pupil_confidence_threshold',.6)
    g_pool.detection_mapping_mode = session_settings.get('detection_mapping_mode','2d')
    g_pool.active_calibration_plugin = None



    def launch_eye_process(eye_id,blocking=False):
        if eyes_are_alive[eye_id].value:
            logger.error("Eye%s process already running."%eye_id)
            return
        launcher_pipe.send(eye_id)
        eye_pipes[eye_id].send( ('Set_Detection_Mapping_Mode',g_pool.detection_mapping_mode) )

        if blocking:
            #wait for ready message from eye to sequentialize startup
            eye_pipes[eye_id].send('Ping')
            eye_pipes[eye_id].recv()

        logger.warning('Eye %s process started.'%eye_id)

    def stop_eye_process(eye_id,blocking=False):
        if eyes_are_alive[eye_id].value:
            eye_pipes[eye_id].send('Exit')
            if blocking:
                while eyes_are_alive[eye_id].value:
                    sleep(.1)


    def set_detection_mapping_mode(new_mode):
        if new_mode == '2d':
            for p in g_pool.plugins:
                if "Vector_Gaze_Mapper" in p.class_name:
                    logger.warning("The gaze mapper is not supported in 2d mode. Please recalibrate.")
                    p.alive = False
            g_pool.plugins.clean()
        for alive, pipe in zip(g_pool.eyes_are_alive,g_pool.eye_pipes):
            if alive.value:
                pipe.send( ('Set_Detection_Mapping_Mode',new_mode) )

        g_pool.detection_mapping_mode = new_mode


    #plugins that are loaded based on user settings from previous session
    g_pool.notifications = []
    g_pool.delayed_notifications = {}
    g_pool.plugins = Plugin_List(g_pool,plugin_by_name,session_settings.get('loaded_plugins',default_plugins))


    if session_settings.get('eye1_process_alive',False):
        launch_eye_process(1,blocking=True)
    if session_settings.get('eye0_process_alive',True):
        launch_eye_process(0,blocking=False)


    ts = get_timestamp()
    # Event loop
    while True:


        #update performace graphs
        t = get_timestamp()
        dt,ts = t-ts,t

        #a dictionary that allows plugins to post and read events
        events = {}

        #report time between now and the last loop interation
        events['dt'] = get_dt()

        #receive and map pupil positions
        recent_pupil_positions = []
        while not g_pool.pupil_queue.empty():
            p = g_pool.pupil_queue.get()
            recent_pupil_positions.append(p)
        events['pupil_positions'] = recent_pupil_positions


        # publish delayed notifiactions when their time has come.
        for n in g_pool.delayed_notifications.values():
            if n['_notify_time_'] < time():
                del n['_notify_time_']
                del g_pool.delayed_notifications[n['subject']]
                g_pool.notifications.append(n)

        # notify each plugin if there are new notifications:
        while g_pool.notifications:
            n = g_pool.notifications.pop(0)
            for p in g_pool.plugins:
                p.on_notify(n)

        # allow each Plugin to do its work.
        for p in g_pool.plugins:
            p.update(frame,events)

        #check if a plugin need to be destroyed
        g_pool.plugins.clean()


    glfw.glfwRestoreWindow(main_window) #need to do this for windows os
    session_settings['loaded_plugins'] = g_pool.plugins.get_initializers()
    session_settings['pupil_confidence_threshold'] = g_pool.pupil_confidence_threshold
    session_settings['gui_scale'] = g_pool.gui.scale
    session_settings['ui_config'] = g_pool.gui.configuration
    session_settings['capture_settings'] = g_pool.capture.settings
    session_settings['window_size'] = glfw.glfwGetWindowSize(main_window)
    session_settings['window_position'] = glfw.glfwGetWindowPos(main_window)
    session_settings['version'] = g_pool.version
    session_settings['eye0_process_alive'] = eyes_are_alive[0].value
    session_settings['eye1_process_alive'] = eyes_are_alive[1].value
    session_settings['detection_mapping_mode'] = g_pool.detection_mapping_mode
    session_settings['audio_mode'] = audio.audio_mode
    session_settings.close()

    # de-init all running plugins
    for p in g_pool.plugins:
        p.alive = False
    g_pool.plugins.clean()


    #shut down eye processes:
    stop_eye_process(0,blocking = True)
    stop_eye_process(1,blocking = True)

    #shut down laucher
    launcher_pipe.send("Exit")

    logger.info("Process Shutting down.")

def world_profiled(pupil_queue,timebase,launcher_pipe,eye_pipes,eyes_are_alive,user_dir,version,cap_src):
    import cProfile,subprocess,os
    from world import world
    cProfile.runctx("world(pupil_queue,timebase,launcher_pipe,eye_pipes,eyes_are_alive,user_dir,version,cap_src)",{'pupil_queue':pupil_queue,'timebase':timebase,'launcher_pipe':launcher_pipe,'eye_pipes':eye_pipes,'eyes_are_alive':eyes_are_alive,'user_dir':user_dir,'version':version,'cap_src':cap_src},locals(),"world.pstats")
    loc = os.path.abspath(__file__).rsplit('pupil_src', 1)
    gprof2dot_loc = os.path.join(loc[0], 'pupil_src', 'shared_modules','gprof2dot.py')
    subprocess.call("python "+gprof2dot_loc+" -f pstats world.pstats | dot -Tpng -o world_cpu_time.png", shell=True)
    print "created cpu time graph for world process. Please check out the png next to the world.py file"