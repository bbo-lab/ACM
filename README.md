# ACM (Anatomically-constrained model)
A framework for for videography based pose tracking of rodents.
By Arne Monsees.

## Installation

### Linux
(Note: Windows support is planned, but currently not present.)

1. [Install Anaconda](https://docs.anaconda.com/anaconda/install/linux/)
2. Clone https://github.com/bbo-lab/ACM.git 
3. Create conda environment `conda env create -f https://raw.githubusercontent.com/bbo-lab/ACM/main/environment.yml`
4. Navigate into the ACM repository
5. Install using `pip install .`

## Testing

1. Download [example dataset](https://www.dropbox.com/sh/040587pwx5t7uh3/AAAI5MVilFrJY-mEPr97uADNa?dl=0).
2. Activate conda environment `conda activate bbo_acm`.
3. (TO BE IMPLEMENTED) View 2d labels by running `python -m ACM --show_input [Path of "table_1_20210511" folder from step 1.]`
4. Run `python -m ACM [Path of "table_1_20210511" folder from step 1.]` (expected to take around 20 mins on modern 8-core CPU - preliminary number from notebook with an AMD Ryzen 7 PRO 5850U, using the iGPU, so probaably notably faster on GPU workstation).
5. (TO BE IMPLEMENTED) View 3d tracked pose by running `python -m ACM --show_result [Path of "table_1_20210511" folder from step 1.]`

## Setting up your own dataset config

### Overview

#### Input

A dataset input config consists of a folder with the following files:

- `configuration.py`: Configuration file that determines settings and input and out paths
- `model.npz`:
- `multicalibration.npy`:
- `origin_coord.npy`: Coordinate system of the final result, relative to the coordinate system of the camera calibration
- `labels_manual.npz`: Manual labels on the videography data
- `labels_dlc_n_m.npy`: Automatically detected labels on the videography data

#### Output

A dataset output consists of a folder with the following files:

- `save_dict.npy`: ...
- `x_calib.npy`: ...
- `x_ini.npy`: ...

By default, results are saved into the `results` subfolder of the datset configuration.

### File structure inputs
#### configuration.py
For standard input and output structure and options, only the start and end frame indices of automatic 2d detection and desired output need to be specified.

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
Manual labels of markers on the videography data in the format
```
Whatever structure this file has
```

#### labels_dlc_n_m.npy
Automatically labels of markers on the videography data in the format
```
Whatever structure this file has
```

### File structure outputs
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
