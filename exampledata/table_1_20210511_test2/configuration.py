# define location of base folder
folder_project = '/home/monsees_la/Dropbox/phd/code/animal_behavior/python__pose_05/share' # make this the location of this file
date = '20210511'
task = 'table_1'
index_frame_start = 72400      
index_frame_end = 73600

# probability cutoff (labels with a smaller probability than pcutoff are not used for the pose reconstruction)
pcutoff = 0.9
# slope of the custom clipping function
slope = 1.0
# initial values of the covariance matrices' diagonal entries
noise = 1e-4

# MODE
mode = 4 # should always be 4

# folder to save pose reconstruction to
folder_save = folder_project + '/data/' + \
              '{:s}_{:s}_test2'.format(task, date)

# define location of all needed files
folder_reqFiles = folder_project + '/required_files/' + date
file_origin_coord = folder_reqFiles + '/' + task + '/origin_coord.npy'
file_calibration = folder_reqFiles + '/' + task + '/multicalibration.npy'
file_model = folder_reqFiles + '/model.npy'
file_labelsDLC = folder_reqFiles + '/' + task + '/labels_dlc_006900_184000.npy'
file_labelsManual = folder_reqFiles + '/' + task + '/labels.npz' # only needed for calibration

# camera calibration scaling factor (calibration board square size -> cm)
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
index_frames_calib = list([[6900, 44300],])
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
use_cuda = False # running code on the CPU generally seems to be faster
slow_mode = True # set to True to lower memory requirements (should not be changed for now)
sigma_point_scheme = 3 # # UKF3: 3, UKF5: 5, naive: 0 (should always be 3)
tol = 5e-2 # tolerance for convergence 
iter_max = 100 # maximum number of EM iterations
