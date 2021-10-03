### File structure inputs
#### configuration.py
For standard input and output structure and options, only the start and end frame indices of automatic 2d detection and desired output need to be specified.
**<-- This is probably not true, as there is e.g. a field body_weight. Please expand!**

#### model.npy

The model in the format
```
dict(
    'surface_vertices':,
    'surface_triangles':,
    'skeleton_vertices':,
    'skeleton_edges':,
    'bone_lengths':,
    'coord0':,
    'skeleton_coords':,
    'skeleton_coords_index':,
    'surface_vertices_weights':,
    'skeleton_vertices_links':,
    'joint_marker_vectors':,
    'joint_marker_distances':,
    'joint_order':,
    'joint_marker_index':,
    'joint_marker_order':,
```

#### multicalibration.npy

The multicalibration in the format
```
dict(
    'recFileNames':,
    'headers':,
    'nCameras':,
    'nFrames':,
    'boardWidth':,
    'boardHeight':,
    'mask_multi':,
    'indexRefCam':,
    'calib':,
    'mask_single':,
    'calib_single':,
    'mask_all':,
    'x0_all':,
    'free_para_all':,
    'tolerance':,
    'message':,
    'convergence':,
    'x_all_fit':,
    'rX1_fit':,
    'RX1_fit':,
    'tX1_fit':,
    'k_fit':,
    'A_fit':,
    'r1_fit':,
    'R1_fit':,
    't1_fit':,
    'r1_single_fit':,
    'R1_single_fit':,
    't1_single_fit'':,
)
```
TODO: Mark/limit to actually necessary fields, to make it possible to build such a file with other software than ours.
A multicam calibration in this format can be produced with our calibration software [calibcam](https://github.com/bbo-lab/calibcam), using charuco boards.

#### origin_coord.npy

Calibrations produced by [calibcam](https://github.com/bbo-lab/calibcam) are aligned based on the camera 1 position and direction. origin_coord determines the coordinate system of the final result, relative to the coordinate system of the camera calibration. This usually has the arena center as origin and the z axis pointing upwards.
```
dict(
    # origin of arena system in the coordinate system of the multicalibration
    'origin': np.ndarray, shape=(3,), dtype('float64'),
    # orientation [lateral1, lateral2, up] of arena system in the coordinate system of the multicalibration
    'coord':, np.ndarray, shape=(3,3), dtype('float64'),
)
```
Thus, `origin_coord:coord * x_arena + origin_coord:origin = x_camera`.

#### labels_manual.npz

Manual labels of markers on the videography data in the format.
Content is a dictionary with integer frame indices as keys (`[frameidx]` below). Each value is another dictionary with the marker names as keys (`[marker name]` below).
```
dict(
    [frameidx]: dict(
        # Positions of markers in 2d image
        [marker name]: np.ndarray, shape=(nCamera,2), dtype('int64'),
        ...
    )
    ...
)
```
In the default model, maker names are 'spot_tail_001', 'spot_tail_002', 'spot_tail_003', 'spot_tail_004', 'spot_tail_005', 'spot_tail_006', 'spot_spine_003', 'spot_spine_004', 'spot_spine_005', 'spot_spine_006', 'spot_head_001', 'spot_head_002', 'spot_head_003', 'spot_shoulder_left', 'spot_shoulder_right', 'spot_side_right', 'spot_hip_right', 'spot_knee_right', 'spot_ankle_right', 'spot_toe_right_001', 'spot_toe_right_002', 'spot_toe_right_003', 'spot_finger_right_001', 'spot_finger_right_002', 'spot_spine_002', 'spot_paw_front_left', 'spot_finger_left_001', 'spot_finger_left_002', 'spot_finger_left_003', 'spot_side_left', 'spot_hip_left', 'spot_toe_left_001', 'spot_toe_left_002', 'spot_spine_001', 'spot_ankle_left', 'spot_paw_hind_left'.
#### labels_dlc_n_m.npy

Automatically labels of markers on the videography data in the format
```
dict(
    # Path of the file itself. TODO: Why is this here? It doesn't seem to be used and
    # it doesn't make a lot of sense to save it's own current path into a file ...
    'file_save':  str,
    # List of frame idx that the frames in "labels_all" correspond to
    'frame_list': np.ndarray, shape=(nFrames,nMarkers,3), dtype('int64'),
    # Positions of markers in 2d image, plus confidence value from dlc
    'labels_all': np.ndarray, shape=(nFrames,nCams,nMarkers,3), dtype('float64'),
)
```
