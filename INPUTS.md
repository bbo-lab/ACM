### File structure inputs

#### configuration.py

# General
For running the example dataset, only the start and end frame indices `index_frame_start` and `index_frame_end` as well as the desired output location `folder_save` need to be specified.

# Calibration
In the calibration step the anatomy of the animal is learned, based on manually annotaded labels. The most important parameters are:
```
# the body weight of the animal in gram
body_weight = 72.0
# the number of frames to skip between time points in the sequence used for learning the animal's antomy
dFrames_calib = 200 
# the start and end frame index of the sequence used for learning the animal's antomy
index_frames_calib = list([[6900, 44300],])
```
Thus, `dFrames_calib=50` and `index_frames_calib=list([[0,100],])` will result in learning the anatomy on the frames with indicex 0, 50 and 100. These frames need to be manually annotated in `labels_manual.npz`.

# Initializaiton
In the initialization step the animal's pose in the first time point of the sequence, which sould be reconstructed, is learnid via gradient descent optimization

# Pose reconstruction
In the final step model parameters are learned via the EM algorithm to allow for effectively reconstructing a sequence with the RTS smoother. The most important parameter is:
```
# the number of frames to skip between time points in the sequence, which sould be reconstructed
dt = 5 
```

#### model.npy

The model defining the skeleton graph of the animal in the format:
```
dict(
    # bone lengths (legacy: can just be filled with zeros) # LEGACY
    'bone_lengths': np.ndarray, shape=(nBones,), dtype('float64'), # LEGACY
    # defines to which joint a marker is connected to (i.e. each element is a respective joint index)
    'joint_marker_index': np.ndarray, shape=(nMarkers,), dtype('int64'),
    # ordered list of marker names
    'joint_marker_order': list, shape=(nMarkers,), type('str'),
    # relative marker locations (legacy: can just be filled with zeros) # LEGACY
    'joint_marker_vectors': np.ndarray, shape=(nMarkers, 3,), dtype('float64'), # LEGACY
    # ordered list of joint names
    'joint_order': list, shape=(nJoints,), type('str'),
    # local bone coordinate systems in rest pose (should be consistent with 'skeleton_vertices')
    'skeleton_coords': np.ndarray, shape=(nBones, 3, 3,), dtype('float64'),
    # bone indices (legacy: can just be filled with zeros) # LEGACY
    'skeleton_coords_index': np.ndarray, shape=(nBones,), dtype('int64'), # LEGACY
    # defines the bones of the skeleton graph (i.e. each entry contains indices of two joints, which form a bone when connected)
    'skeleton_edges': np.ndarray, shape=(nBones, 2,), dtype('int64'),
    # absolute joint positions in rest pose (should be consistent with 'skeleton_coords'; resulting bone lengths should all be equal to one)
    'skeleton_vertices': np.ndarray, shape=(nJoints, 3,), dtype('float64'),
    # indicates if a bone is affected by a rotation of a preceding bone in the skeleton graph
    'skeleton_vertices_links': np.ndarray, shape=(nBones, nBones,), dtype('bool'),
    # defines triangles of a surface mesh (i.e. each triagle is defined by three vertex indices) (legacy: can just be an empty array) # LEGACY
    'surface_triangles': np.ndarray, shape=(nTriangles, 3,), dtype('int32'), # LEGACY
    # defines absolute locations of vertices of a surface mesh  (legacy: can just be an empty array) # LEGACY
    'surface_vertices': np.ndarray, shape=(nVertices, 3,), dtype('float64'), # LEGACY
    # defines weights of surface vertices such that they can be adjusted according to linear blend skinning (legacy: can just be an empty array) # LEGACY
    'surface_vertices_weights': np.ndarray, shape=(nVertices, nJoints,), dtype('float64'), # LEGACY
)
```
Any additional keys of the dictionary are not mandatory and originate from the development phase of the method.
In the default model, joint names are: 'joint_ankle_left', 'joint_ankle_right', 'joint_elbow_left', 'joint_elbow_right', 'joint_finger_left_002', 'joint_finger_right_002', 'joint_head_001', 'joint_hip_left', 'joint_hip_right', 'joint_knee_left', 'joint_knee_right', 'joint_paw_hind_left', 'joint_paw_hind_right', 'joint_shoulder_left', 'joint_shoulder_right', 'joint_spine_001', 'joint_spine_002', 'joint_spine_003', 'joint_spine_004', 'joint_spine_005', 'joint_tail_001', 'joint_tail_002', 'joint_tail_003', 'joint_tail_004', 'joint_tail_005', 'joint_toe_left_002', 'joint_toe_right_002', 'joint_wrist_left', 'joint_wrist_right'
In the default model, marker names are: 'marker_ankle_left_start', 'marker_ankle_right_start', 'marker_elbow_left_start', 'marker_elbow_right_start', 'marker_finger_left_001_start', 'marker_finger_left_002_start', 'marker_finger_left_003_start', 'marker_finger_right_001_start', 'marker_finger_right_002_start', 'marker_finger_right_003_start', 'marker_head_001_start', 'marker_head_002_start', 'marker_head_003_start', 'marker_hip_left_start', 'marker_hip_right_start', 'marker_knee_left_start', 'marker_knee_right_start', 'marker_paw_front_left_start', 'marker_paw_front_right_start', 'marker_paw_hind_left_start', 'marker_paw_hind_right_start', 'marker_shoulder_left_start', 'marker_shoulder_right_start', 'marker_side_left_start', 'marker_side_right_start', 'marker_spine_001_start', 'marker_spine_002_start', 'marker_spine_003_start', 'marker_spine_004_start', 'marker_spine_005_start', 'marker_spine_006_start', 'marker_tail_001_start', 'marker_tail_002_start', 'marker_tail_003_start', 'marker_tail_004_start', 'marker_tail_005_start', 'marker_tail_006_start', 'marker_toe_left_001_start', 'marker_toe_left_002_start', 'marker_toe_left_003_start', 'marker_toe_right_001_start', 'marker_toe_right_002_start', 'marker_toe_right_003_start'.
Currently a few joint names are hard coded. Therefore changing these names is not recommended.
The hard coded joint names are: 'joint_hip_left', 'joint_hip_right', 'joint_shoulder_left', 'joint_shoulder_left' (see functions `get_coord0` and `get_skeleton_coords0` in `anatomy.py`).

#### multicalibration.npy

The multicalibration in the format:
```
dict(
    # focal lengths and principal point locations for each camera (intrinsic parameters)
    'A_fit': np.ndarray, shape=(nCameras, 4,), dtype('float64'),
    # rotation matrix for each camera (extrinsic parameters)
    'RX1_fit': np.ndarray, shape=(nCameras, 3, 3,), dtype('float64'),
    # distortion coefficients for each camera (intrinsic parameters)
    'k_fit': np.ndarray, shape=(nCameras, 5,), dtype('float64'),
    # total number of calibrated cameras
    'nCameras': type('int'),
    # Rodrigues vectors corresponding to 'RX1_fit' for each camera (extrinsic parameters)
    'rX1_fit': np.ndarray, shape=(nCameras, 3,), dtype('float64'),
    # translation vector for each camera (extrinsic parameters)
    'tX1_fit': np.ndarray, shape=(nCameras, 3,), dtype('float64'),
)
```
Any additional keys of the dictionary are not mandatory and originate from the development phase of the method.
A multi-camera calibration file in this format can be produced with our calibration software [calibcam](https://github.com/bbo-lab/calibcam), using ChArUco boards.

#### origin_coord.npy

Multicalibrations produced by [calibcam](https://github.com/bbo-lab/calibcam) are aligned based on the camera 1 position and direction. origin_coord determines the coordinate system of the final result, relative to the coordinate system of the camera calibration. This usually has the arena center as origin and the z-axis pointing upwards. The format is:
```
dict(
    # origin of the arena coordinate system in the coordinate system of the multicalibration
    'origin': np.ndarray, shape=(3,), dtype('float64'),
    # orientation of arena coordinate system in the coordinate system of the multicalibration (i.e. [lateral1, lateral2, up])
    'coord': np.ndarray, shape=(3, 3,), dtype('float64'),
)
```
Thus, `origin_coord:coord * x_arena + origin_coord:origin = x_camera`.

#### labels_manual.npz

Manual labels of surface markers on the videography data. Content is a dictionary with integer frame indices as keys (`[frameidx]` below). Each value is another dictionary with the label names as keys (`[label name]` below). The format is:
```
dict(
    [frameidx]: dict(
        # positions of labels in 2D image
        [label name]: np.ndarray, shape=(nCameras, 2), dtype('int64'),
        ...
    )
    ...
)
```
In the default model, labels names are: 'spot_ankle_left', 'spot_ankle_right', 'spot_elbow_left', 'spot_elbow_right', 'spot_finger_left_001', 'spot_finger_left_002', 'spot_finger_left_003', 'spot_finger_right_001', 'spot_finger_right_002', 'spot_finger_right_003', 'spot_head_001', 'spot_head_002', 'spot_head_003', 'spot_hip_left', 'spot_hip_right', 'spot_knee_left', 'spot_knee_right', 'spot_paw_front_left', 'spot_paw_front_right', 'spot_paw_hind_left', 'spot_paw_hind_right', 'spot_shoulder_left', 'spot_shoulder_right', 'spot_side_left', 'spot_side_right', 'spot_spine_001', 'spot_spine_002', 'spot_spine_003', 'spot_spine_004', 'spot_spine_005', 'spot_spine_006', 'spot_tail_001', 'spot_tail_002', 'spot_tail_003', 'spot_tail_004', 'spot_tail_005', 'spot_tail_006', 'spot_toe_left_001', 'spot_toe_left_002', 'spot_toe_left_003', 'spot_toe_right_001', 'spot_toe_right_002', 'spot_toe_right_003'.
To differentiate manually annotated labels from reconstructed marker locations, these names are later on modified internally (see `joint_marker_order` in `model.npy`).
Currently a few label names are hard coded. Therefore changing these names is not recommended (see function `initialize_x` in `calibration.py`).
The hard coded label names are: 'spot_head_002', 'spot_head_003'. Labels with the strings 'spine' and 'tail' in their names are used for roughly aligning the 3D orientation of the animal before learning its anatomy.

#### labels_dlc_n_m.npy

Automatically detected labels of markers on the videography data in the format:
```
dict(
    # original file location (legacy: can just be an empty string) # LEGACY
    'file_save', type('str') # LEGACY
    # list of frame indices that the frames in 'labels_all' correspond to
    'frame_list': np.ndarray, shape=(nFrames,), dtype('int64'),
    # positions of markers in 2D image plus confidence value from DLC
    'labels_all': np.ndarray, shape=(nFrames, nCameras, nMarkers, 3,), dtype('float64'),
)
```
Any additional keys of the dictionary are not mandatory and originate from the development phase of the method.