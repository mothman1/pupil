
import cPickle as pickle
import os

pl_c = None
file_path = os.path.expanduser('/Developments/NCLUni/pupil_crowd4Jul16/recordings/2016_07_29/003_J13J14/CrowdSSMIAutoPupil/pupil_data')
with open(file_path,'rb') as fh:
    data = fh.read()
    pl_c = pickle.loads(data)
# pl_c = None
# file_path = os.path.expanduser('/Developments/NCLUni/pupil_crowd4Jul16/recordings/2016_07_29/003Crowd2CircleCursor/pupil_data')
# with open(file_path,'rb') as fh:
#     data = fh.read()
#     pl_c = pickle.loads(data)

import matplotlib.pyplot as plt

plc_g_center_y = [(gx['norm_pos'][1] * 720) for gx in pl_c['gaze_positions']]
plc_g_center_x = [(gx['norm_pos'][0] * 1280) for gx in pl_c['gaze_positions']]
plt.plot(plc_g_center_x, plc_g_center_y, 'ro')
plt.axis([0, 1280, 0, 720])


plc_p_center_y = [(px['norm_pos'][1] * 480) for px in pl_c['pupil_positions']]
plc_p_center_x = [(px['norm_pos'][0] * 640) for px in pl_c['pupil_positions']]
plt.plot(plc_p_center_x, plc_p_center_y, 'bo')
plt.axis([0, 640, 0, 480])
plt.show()