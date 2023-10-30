#!/usr/bin/env python3

import importlib
import numpy as np
import os
from scipy import io
import sys
import torch

def export(result_path):
    folder_save = result_path
    project_path = os.path.abspath(result_path+"/../..")

    sys.path.append(project_path)
    print(project_path)
    import configuration as cfg
    from ACM import anatomy
    from ACM import helper
    from ACM import model
    from ACM import routines_math as rout_m
    sys.path.pop(sys.path.index(project_path))

    if not hasattr(cfg,'animal_is_large'):
        cfg.animal_is_large = False

    list_is_large_animal = [cfg.animal_is_large]
    
    importlib.reload(anatomy)
        
    # get arguments
    file_origin_coord = cfg.file_origin_coord
    file_calibration = cfg.file_calibration
    file_model = cfg.file_model
    file_labelsDLC = cfg.file_labelsDLC

    args_model = helper.get_arguments(file_origin_coord, file_calibration, file_model, file_labelsDLC,
                                        cfg.scale_factor, cfg.pcutoff)
    
    args_model['use_custom_clip'] = True
    nBones = args_model['numbers']['nBones']
    nMarkers = args_model['numbers']['nMarkers']
    #
    joint_order = args_model['model']['joint_order']
    skel_edges = args_model['model']['skeleton_edges'].cpu().numpy()
    skel_coords0 = args_model['model']['skeleton_coords0'].cpu().numpy()
    bone_lengths_index = args_model['model']['bone_lengths_index'].cpu().numpy()

    # get save_dict
    save_dict = np.load(result_path+'/save_dict.npy', allow_pickle=True).item()
    if ('mu_uks' in save_dict):
        mu_uks_norm = np.copy(save_dict['mu_uks'][1:])
    else:
        mu_uks_norm = np.copy(save_dict['mu_fit'][1:])
    mu_uks = model.undo_normalization(torch.from_numpy(mu_uks_norm), args_model).numpy() # reverse normalization

    # get x_ini
    x_ini = np.load(result_path+'/x_ini.npy', allow_pickle=True)

    # free parameters
    free_para_pose = args_model['free_para_pose'].cpu().numpy()
    free_para_bones = np.zeros(nBones, dtype=bool)
    free_para_markers = np.zeros(nMarkers*3, dtype=bool)    
    free_para = np.concatenate([free_para_bones,
                                free_para_markers,
                                free_para_pose], 0)

    # full x
    nT_use = np.size(mu_uks_norm, 0)
    x = np.tile(x_ini, nT_use).reshape(nT_use, len(x_ini))
    x[:, free_para] = mu_uks

    # requested parameters
    nPara_bones = args_model['nPara_bones']
    nPara_markers = args_model['nPara_markers']
    nPara_skel = nPara_bones + nPara_markers
    #
    bone_lengths = x_ini[:nPara_bones] # normalization in x_ini already reversed
    bone_lengths = bone_lengths[bone_lengths_index] # fill bone lengths of 'right' bones with the values from the 'left' bones
    #
    t0 = x[:, nPara_skel:nPara_skel+3].reshape(nT_use, 3)
    r = x[:, nPara_skel+3:].reshape(nT_use, nBones, 3)
    R = np.zeros((nT_use, nBones, 3, 3), dtype=np.float64)
    for i in range(nT_use):
        for j in range(nBones):
            R[i, j] = rout_m.rodrigues2rotMat_single(r[i, j])

    # dict to be saved
    data_dict = dict()
    data_dict['joint_names'] = joint_order
    data_dict['edges'] = skel_edges
    data_dict['coords0'] = skel_coords0
    data_dict['bone_lengths'] = bone_lengths
    data_dict['t'] = t0
    data_dict['R'] = R
    data_dict['index_frame_start'] = cfg.index_frame_start
    data_dict['origin'] = np.load(file_origin_coord,allow_pickle=True).item()
    data_dict['folder_ccv'] = cfg.folder_video
    
    # save
    np.savez(folder_save+'/motiondata.npz', data_dict)
    io.savemat(folder_save+'/motiondata.mat', data_dict)
        
if __name__ == '__main__':
    export(sys.argv[1])
