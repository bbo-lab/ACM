#!/usr/bin/env python3

import os
import numpy as np
import torch
from scipy.io import savemat

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
    
     # create correct free_para
    free_para = np.concatenate([free_para_bones,
                                free_para_markers,
                                free_para_pose], 0)   
    # load x_ini
    x_ini = np.load(cfg.folder_save + '/x_ini.npy', allow_pickle=True)
    x_ini_free = x_ini[free_para]
    
    # update args regarding fixed tensors
    args['plot'] = False
    args['nFrames'] = int(1)
    
    # update args regarding x0 and labels
    # ARGS X
    args['x_torch'] = torch.from_numpy(x_ini)
    args['x_free_torch'] = torch.from_numpy(x_ini[free_para])
    args['x_free_torch'].requires_grad = True
    # ARGS LABELS DLC
    labels_dlc = np.load(cfg.file_labelsDLC, allow_pickle=True).item()
    i_frame_single = np.where(cfg.index_frame_ini == labels_dlc['frame_list'])[0][0]
    args['labels_single_torch'] = args['labels'][i_frame_single][None, :].clone()
    args['labels_mask_single_torch'] = args['labels_mask'][i_frame_single][None, :].clone()

    # BOUNDS
    # pose
    bounds_free_pose = args['bounds_free_pose']
    bounds_free_low_pose = model.do_normalization(bounds_free_pose[:, 0][None, :], args).numpy().ravel()
    bounds_free_high_pose = model.do_normalization(bounds_free_pose[:, 1][None, :], args).numpy().ravel()
    bounds_free = np.stack([bounds_free_low_pose, bounds_free_high_pose], 1)
    args['bounds_free'] = bounds_free
            
    # normalize x
    x_ini_free_norm = model.do_normalization(torch.from_numpy(x_ini_free.reshape(1, nFree_pose)), args).numpy().ravel()
    
    # OPTIMIZE
    # create optimization dictonary
    opt_options = dict()
    opt_options['disp'] = cfg.opt_options_fit__disp
    opt_options['maxiter'] = cfg.opt_options_fit__maxiter
    opt_options['maxcor'] = cfg.opt_options_fit__maxcor
    opt_options['ftol'] = cfg.opt_options_fit__ftol
    opt_options['gtol'] = cfg.opt_options_fit__gtol
    opt_options['maxfun'] = cfg.opt_options_fit__maxfun
    opt_options['iprint'] = cfg.opt_options_fit__iprint
    opt_options['maxls'] = cfg.opt_options_fit__maxls
    opt_dict = dict()
    opt_dict['opt_method'] = cfg.opt_method_fit
    opt_dict['opt_options'] = opt_options  
    
    # optimize
    x_fit_free_norm__all = np.zeros((cfg.nT+1, nFree_pose), dtype=np.float64)
    x_fit_free_norm__all[0, :] = np.nan # corresponds to mu0
    x_previous = np.copy(x_ini)
    x_previous_free_norm = np.copy(x_ini_free_norm)
    x_fit_free_norm__all__counter = 1
    for i in np.arange(cfg.nT):
        # PRINT
        print('Optimizing frame {:09d} / {:09d}'.format(cfg.index_frame_ini + i * cfg.dt, cfg.index_frame_ini + cfg.nT * cfg.dt - 1))
        
        # update args regarding x0 and labels
        # ARGS X
        args['x_torch'].data.copy_(torch.from_numpy(x_previous).data)
        args['x_free_torch'].data.copy_(torch.from_numpy(x_previous[free_para]).data)
        # ARGS LABELS DLC
        i_frame_single = np.where((cfg.index_frame_ini + i * cfg.dt) == labels_dlc['frame_list'])[0][0]
        args['labels_single_torch'].data.copy_(args['labels'][i_frame_single][None, :].data)
        args['labels_mask_single_torch'].data.copy_(args['labels_mask'][i_frame_single][None, :].data)
        
        # optimize
        min_result = opt.optimize__scipy(x_previous_free_norm, args,
                                         opt_dict)
        x_fit_free_norm = np.copy(min_result.x)
        x_fit_free = model.undo_normalization(torch.from_numpy(x_fit_free_norm).reshape(1, nFree_pose), args).numpy().ravel()
                
        x_fit_free_norm__all[i+1] = np.copy(x_fit_free_norm)
        
        x_previous_free_norm = np.copy(x_fit_free_norm)
        x_previous[free_para] = np.copy(x_fit_free)

    # save    
    save_dict = dict()
    save_dict['mu_fit'] = x_fit_free_norm__all
    np.save(os.path.join(cfg.folder_save,'save_dict.npy'), save_dict)
    
    # to get 3D joint & marker locations
    args['plot'] = True
    marker_proj, marker_pos, skel_pos = model.fcn_emission_free(torch.from_numpy(x_fit_free_norm__all[1:]), args)
    marker_proj = marker_proj.detach().cpu().numpy().reshape(cfg.nT, nCameras, nMarkers, 2)
    marker_pos = marker_pos.detach().cpu().numpy()
    skel_pos = skel_pos.detach().cpu().numpy()
    pose = dict()
    pose['marker_positions_2d'] = marker_proj
    pose['marker_positions_3d'] = marker_pos
    pose['joint_positions_3d'] = skel_pos

    posepath = os.path.join(cfg.folder_save,'pose')
    print(f'Saving pose to {posepath}')
    np.save(posepath+'.npy', pose)
    savemat(posepath+'.mat', pose)
    
if __name__ == "__main__":
    main()