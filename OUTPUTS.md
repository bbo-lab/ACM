### File structure outputs
TODO: A dictionary with the configuration should also be saved, including a code version number (pip install) or a git hash (dev version)

#### pose.npy

```
dict(
    # Positions of markers in 2d image TODO: Is this a section of labels_dlc:labels_all?
    'marker_positions_2d': np.ndarray, shape=(nFrames,nCams,nMarkers,2), dtype('float64'),
    # Marker position in 3d TODO: some words about inference
    'marker_positions_3d': np.ndarray, shape=(nFrames,nMarkers,3), dtype('float64'),
    # Joint position in 3d TODO: some words about inference
    'joint_positions_3d':  np.ndarray, shape=(nFrames,nJoints,3), dtype('float64'),
)
```
TODO: For easily working with this, it is missing a frame_list field. Currently needs to be taken from the configuration.

#### save_dict.npy

```
Whatever structure this file has
```

#### `x_calib.npy

```
Whatever structure this file has
```

#### `x_ini.npy

```
Whatever structure this file has
```
