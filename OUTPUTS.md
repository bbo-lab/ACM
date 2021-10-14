### File structure outputs

#### x_calib.npy

Learned bone lengths and relative marker positions as well as pose variables (i.e. bone rotations and a translations) for all time points of the sequence, which is used for learning the animal's anatomy. The format is:
```
np.ndarray, shape=(nBones + 3*nMarkers + nFrames_calib*(3*nBones+3),), dtype('float64'),
```

#### x_ini.npy

Learned bone lengths and relative marker positions as well as pose variables (i.e. bone rotations and a single translation) for the first time point of the sequence, which should be reconstructed. The format is:
```
np.ndarray, shape=(nBones + 3*nMarkers + 3*nBones+3,), dtype('float64'),
```

#### save_dict.npy

Learned model parameters as well as resulting latent variables and corresponding covariance matrices for the entire sequence. The format is:
```
dict(
    # learned transition matrix (legacy: this is fixed to the identity matrix) # LEGACY
    'A': , np.ndarray, shape=(nLatent, nLatent,), dtype('float64'), # LEGACY
    # learned inital state of the latent variables
    'mu0': np.ndarray, shape=(nLatent,), dtype('float64'),
    # inferred latent variables (inferrence is based on the learned model parameters)
    'mu_uks': np.ndarray, shape=(nFrames+1, nLatent,), dtype('float64'),
    # learned covariance matrix of the latent variables
    'var0': np.ndarray, shape=(nLatent, nLatent,), dtype('float64'),
    # learned covariance matrix of the transition noise
    'var_f': np.ndarray, shape=(nLatent, nLatent,), dtype('float64'),
    # learned covariance matrix of the measurement noise
    'var_g': np.ndarray, shape=(nMeasurement, nMeasurement,), dtype('float64'),
    # inferred covariance matirces of the latent variables (inferrence is based on the learned model parameters)
    'var_uks': np.ndarray, shape=(nFrames+1, nMeasurement, nMeasurement,), dtype('float64'),
)
```

#### pose.npy

Inferred 3D joint and marker positions as well as resulting projected marker locations in the 2D images. The format is:
```
dict(
    # reconstructed marker locations in 2D image
    'marker_positions_2d': np.ndarray, shape=(nFrames, nCameras, nMarkers, 2,), dtype('float64'),
    # reconstructed marker positions in 3D
    'marker_positions_3d': np.ndarray, shape=(nFrames, nMarkers, 3,), dtype('float64'),
    # reconstructed joint positions in 3D
    'joint_positions_3d':  np.ndarray, shape=(nFrames, nJoints, 3,), dtype('float64'),
)
```