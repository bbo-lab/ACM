#!/usr/bin/env python3

import numpy as np
import sys
import torch

import configuration as cfg

from . import calibration as calib
from . import helper
from . import model
from . import optimization as opt

def main():
    # get arguments
    args = helper.get_arguments(cfg.file_origin_coord, cfg.file_calibration, cfg.file_model, cfg.file_labelsDLC,
                                cfg.scale_factor, cfg.pcutoff)
    args['use_custom_clip'] = False
    
    # get relevant information from arguments
    nBones = args['numbers']['nBones']
    nMarkers = args['numbers']['nMarkers']
    nCameras = args['numbers']['nCameras']
    joint_order = args['model']['joint_order'] # list
    joint_marker_order = args['model']['joint_marker_order'] # list
    skeleton_edges = args['model']['skeleton_edges'].cpu().numpy()
    bone_lengths_index = args['model']['bone_lengths_index'].cpu().numpy()
    joint_marker_index = args['model']['joint_marker_index'].cpu().numpy()
    #
    free_para_bones = args['free_para_bones'].cpu().numpy()
    free_para_markers = args['free_para_markers'].cpu().numpy()
    free_para_pose = args['free_para_pose'].cpu().numpy()
    nPara_bones = args['nPara_bones']
    nPara_markers = args['nPara_markers']
    nPara_pose = args['nPara_pose']
    nFree_pose = args['nFree_pose']

    # remove all free parameters that do not modify the pose
    free_para_bones = np.zeros_like(free_para_bones, dtype=bool)
    free_para_markers = np.zeros_like(free_para_markers, dtype=bool)
    nFree_bones = int(0)
    nFree_markers = int(0)
    args['free_para_bones'] = torch.from_numpy(free_para_bones)
    args['free_para_markers'] = torch.from_numpy(free_para_markers)
    args['nFree_bones'] = nFree_bones
    args['nFree_markers'] = nFree_markers

    # get index of initialization frame
    labelsDLC = np.load(cfg.file_labelsDLC, allow_pickle=True).item()
    frame_list = labelsDLC['frame_list']
    labels_dlc = labelsDLC['labels_all']
    
    # initialize x_pose
    # load calibrated model and initalize the pose
    x_calib = np.load(cfg.folder_calib + '/x_calib.npy', allow_pickle=True)
    x_bones = x_calib[:nPara_bones]
    x_markers = x_calib[nPara_bones:nPara_bones+nPara_markers]
    # load arena coordinate system
    origin, coord = calib.get_origin_coord(cfg.file_origin_coord, cfg.scale_factor)
    #
    i_frame = np.where(frame_list == cfg.index_frame_ini)[0][0]
    labels_frame = np.copy(labels_dlc[i_frame])
    labels_frame_mask = (labels_frame[:, :, 2] < cfg.pcutoff)
    labels_frame[labels_frame_mask] = 0.0
    x_pose = calib.initialize_x(args,
                                labels_frame,
                                coord, origin)
    #
    x = np.concatenate([x_bones,
                        x_markers,
                        x_pose], 0)
    # create correct free_para
    free_para = np.concatenate([free_para_bones,
                                free_para_markers,
                                free_para_pose], 0)
    
    # update args regarding fixed tensors
    args['plot'] = False
    args['nFrames'] = int(1)
    
    # update args regarding x0 and labels
    # ARGS X
    args['x_torch'] = torch.from_numpy(np.concatenate([x_bones,
                                                       x_markers,
                                                       x_pose], 0))
    args['x_free_torch'] = torch.from_numpy(x_pose[free_para_pose])
    args['x_free_torch'].requires_grad = True
    # ARGS LABELS DLC
    labels_dlc = np.load(cfg.file_labelsDLC, allow_pickle=True).item()
    i_frame_single = np.where(cfg.index_frame_ini == labels_dlc['frame_list'])[0][0]
    args['labels_single_torch'] = args['labels'][i_frame_single][None, :].clone()
    args['labels_mask_single_torch'] = args['labels_mask'][i_frame_single][None, :].clone()

    # BOUNDS
    # pose
    if ((cfg.mode == 1) or (cfg.mode == 2)):
        bounds_free_pose = args['bounds_free_pose']
        bounds_free_low_pose = bounds_free_pose[:, 0]
        bounds_free_high_pose = bounds_free_pose[:, 1]
    elif ((cfg.mode == 3) or (cfg.mode == 4)): 
        # so that pose-encoding parameters do not get initialized with basically infinity when EM algorithm is used
        mu_ini_fac = 0.9
        bounds_free_low_pose = args['bounds_free_pose_0'] - args['bounds_free_pose_range'] * mu_ini_fac
        bounds_free_high_pose = args['bounds_free_pose_0'] + args['bounds_free_pose_range'] * mu_ini_fac
    bounds_free_low_pose = model.do_normalization(bounds_free_low_pose[None, :], args).numpy().ravel()
    bounds_free_high_pose = model.do_normalization(bounds_free_high_pose[None, :], args).numpy().ravel()
    bounds_free = np.stack([bounds_free_low_pose, bounds_free_high_pose], 1)
    args['bounds_free'] = bounds_free
            
    # normalize x
    x_free = model.do_normalization(torch.from_numpy(x_pose[free_para_pose].reshape(1, nFree_pose)), args).numpy().ravel()

    # OPTIMIZE
    # create optimization dictonary
    opt_options = dict()
    opt_options['disp'] = cfg.opt_options_ini__disp
    opt_options['maxiter'] = cfg.opt_options_ini__maxiter
    opt_options['maxcor'] = cfg.opt_options_ini__maxcor
    opt_options['ftol'] = cfg.opt_options_ini__ftol
    opt_options['gtol'] = cfg.opt_options_ini__gtol
    opt_options['maxfun'] = cfg.opt_options_ini__maxfun
    opt_options['iprint'] = cfg.opt_options_ini__iprint
    opt_options['maxls'] = cfg.opt_options_ini__maxls
    opt_dict = dict()
    opt_dict['opt_method'] = cfg.opt_method_ini
    opt_dict['opt_options'] = opt_options  
    print('Initializing')
    min_result = opt.optimize__scipy(x_free, args,
                                     opt_dict)
    print('Finished initializing')
    print()

    # copy fitting result into correct arrary
    x_fit_free = np.copy(min_result.x)
    
    # reverse normalization of x
    x_fit_free = model.undo_normalization(torch.from_numpy(x_fit_free).reshape(1, nFree_pose), args).numpy().ravel()
    
    # add free variables
    x_ini = np.copy(x)
    x_ini[free_para] = x_fit_free
    
    # save
    np.save(cfg.folder_init + '/x_ini.npy', x_ini)

if __name__ == "__main__":
    main()
