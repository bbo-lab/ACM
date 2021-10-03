### File structure inputs
#### configuration.py
For standard input and output structure and options, only the start and end frame indices of automatic 2d detection and desired output need to be specified.
**<-- This is probably not true, as there is e.g. a field body_weight. Please expand!**

#### model.npz

The model in the format
```
Whatever structure this file has
```

#### multicalibration.npy

The multicalibration in the format
```
Whatever structure this file has
```
A multicam calibration in this format can be produced with our calibration software [calibcam](https://github.com/bbo-lab/calibcam), using charuco boards.

#### origin_coord.npz

Calibrations produced by [calibcam](https://github.com/bbo-lab/calibcam) are aligned based on the camera 1 position and direction. origin_coord determines the coordinate system of the final result, relative to the coordinate system of the camera calibration. This usually has the arena center as origin and te z axis pointing upwards.
```
Whatever structure this file has
```

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
