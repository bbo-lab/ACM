import os
from datetime import datetime

# Frames present in the labels file, used to generate file name
dlc_index_frame_start = 6900
dlc_index_frame_end = 184000

# Frames to actuallly process
index_frame_start = 45800
index_frame_end = 46800

# MODE
# 1: deterministic model
# 2: deterministic model + joint angle limits
# 3: probabilistic model
# 4: probabilistic model + joint angle limits
mode = 4 # should always be 4 for now

# probability cutoff (labels with a smaller probability than pcutoff are not used for the pose reconstruction)
pcutoff = 0.9
# slope of the custom clipping function
slope = 1.0
# initial values of the covariance matrices' diagonal entries
noise = 1e-4

# videos [optional, for viewer only]
videos = [ 'cam1_20210511_table_10.ccv.mp4',  'cam2_20210511_table_10.ccv.mp4',  'cam3_20210511_table_10.ccv.mp4',  'cam4_20210511_table_10.ccv.mp4' ]

# camera calibration scaling factor (calibration board square size -> cm) 
# TODO This should really be in the calibration file
scale_factor = 6.5 # [cm]
# whether to use rodrigues parameterization for all joint angles
use_rodrigues = True # should always be True
# whether to use reparameterization of joint angles
use_reparameterization = False # should always be False
# numerical tolerance
num_tol = 2**-52

# NORMALIZATION
normalize_bone_lengths = 1.0 # e.g. 1.0 [cm]
normalize_joint_marker_vec = 1.0 # e.g. 1.0 [cm]
normalize_t0 = 50.0 # e.g. distance from arena origin to furthest arena boundary [cm]
normalize_r0 = 1.5707963267948966 # e.g. pi/2 [rad]
normalize_camera_sensor_x = 1280.0 # e.g. camera sensor size (width) [px]
normalize_camera_sensor_y = 1024.0 # e.g. camera sensor size (height) [px]

# CALIBRATION
plot_calib = False
sigma_factor = 10.0
body_weight = 72.0 # [g]
dFrames_calib = 200 
index_frames_calib = 'all'
#
opt_method_calib = 'L-BFGS-B'
opt_options_calib__disp = False
opt_options_calib__ftol = 2**-23 # scipy default value: 2.220446049250313e-09
opt_options_calib__gtol = 0.0 # scipy default value: 1e-05
opt_options_calib__maxiter = float('inf')
opt_options_calib__maxcor = 100 # scipy default value: 10
opt_options_calib__maxfun = float('inf')
opt_options_calib__iprint = -1
opt_options_calib__maxls = 200 # scipy default value: 20

# INITIALIZATION
plot_ini = False
index_frame_ini = index_frame_start
#
opt_method_ini = 'L-BFGS-B'
opt_options_ini__disp = False
opt_options_ini__ftol = 2**-23 # scipy default value: 2.220446049250313e-09
opt_options_ini__gtol = 0.0 # scipy default value: 1e-05
opt_options_ini__maxiter = float('inf')
opt_options_ini__maxcor = 100 # scipy default value: 10
opt_options_ini__maxfun = float('inf')
opt_options_ini__iprint = -1
opt_options_ini__maxls = 200 # scipy default value: 20

# POSE RECONSTRUCTION
plot_recon = False
dt = 5 # number of time points to skip between frames
nT = int((index_frame_end - index_frame_start) / dt) # number of time points to be reconstruced
# for mode 1 & 2:
opt_method_fit = opt_method_ini
opt_options_fit__disp = False
opt_options_fit__ftol = 2**-23 # scipy default value: 2.220446049250313e-09
opt_options_fit__gtol = opt_options_ini__gtol # scipy default value: 1e-05
opt_options_fit__maxiter = opt_options_ini__maxiter
opt_options_fit__maxcor = opt_options_ini__maxcor
opt_options_fit__maxfun = opt_options_ini__maxfun
opt_options_fit__iprint = opt_options_ini__iprint
opt_options_fit__maxls = opt_options_ini__maxls
# for mode 3 & 4:
use_cuda = False # running code on the CPU generally seems to be faster
slow_mode = True # set to True to lower memory requirements (should not be changed for now)
sigma_point_scheme = 3 # # UKF3: 3, UKF5: 5, naive: 0 (should always be 3)
tol = 5e-2 # tolerance for convergence 
iter_max = 100 # maximum number of EM iterations

# ASSERTS
assert dlc_index_frame_start<=index_frame_start and dlc_index_frame_end>=index_frame_end, "Requested frame range not in label range."

# DEFINE PATHS
folder_project = os.path.dirname(os.path.realpath(__file__))
job_name = os.path.basename(folder_project)
job_time = datetime.now()

# folder to save initialization, calibration and pose reconstruction to
folder_save = os.path.join(folder_project, 'results', f'{job_name}_{job_time.strftime("%Y%m%d-%H%M%S")}')
folder_init = folder_save
folder_calib = folder_save

# add path to video filenames
videos = [ os.path.join(folder_project,video) for video in videos ]

# define location of base folder
folder_reqFiles = folder_project

# define location of all needed files
file_origin_coord = os.path.realpath(os.path.join(folder_reqFiles, 'origin_coord.npy'))
file_calibration = os.path.realpath(os.path.join(folder_reqFiles, 'multicalibration.npy'))
file_model = os.path.realpath(os.path.join(folder_reqFiles, 'model.npy'))
file_labelsDLC = os.path.realpath(os.path.join(folder_reqFiles,  f'labels_dlc_{dlc_index_frame_start:06}_{dlc_index_frame_end:06}.npy'))
file_labelsManual = os.path.realpath(os.path.join(folder_reqFiles, 'labels_manual.npz')) # only needed for calibration
