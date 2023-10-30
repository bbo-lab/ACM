#!/usr/bin/env python3

import numpy as np
import sys
import torch

import configuration as cfg

from . import anatomy
from . import helper
from . import interp_3d
from . import model
from . import optimization as opt
from . import routines_math as rout_m

def get_origin_coord(file_origin_coord, scale_factor):
    # load
    origin_coord = np.load(file_origin_coord, allow_pickle=True).item()
    # arena coordinate system
    origin = origin_coord['origin']
    coord = origin_coord['coord']
    # scaling (calibration board square size -> cm)
    origin = origin * scale_factor
    return origin, coord

# ATTENTION: this is hard coded (assumes specific naming of the surface markers)
def initialize_x(args,
                 labels,
                 arena_coord, arena_origin):
    # get numbers from args
    numbers = args['numbers']
    nCameras = numbers['nCameras'] # int
    nBones = numbers['nBones'] # int
    nMarkers = numbers['nMarkers'] # int
    
    # calibration
    calibration = args['calibration']
    A_entries = calibration['A_fit'].cpu().numpy()
    k = calibration['k_fit'].cpu().numpy()
    rX1 = calibration['rX1_fit'].cpu().numpy()
    RX1 = calibration['RX1_fit'].cpu().numpy()
    tX1 = calibration['tX1_fit'].cpu().numpy()
    A = np.zeros((nCameras, 3, 3), dtype=np.float64)

    if len(A_entries.shape) == 2:  # Old style calibration
        for i_cam in range(nCameras):
            A[i_cam, 0, 0] = A_entries[i_cam, 0]
            A[i_cam, 0, 2] = A_entries[i_cam, 1]
            A[i_cam, 1, 1] = A_entries[i_cam, 2]
            A[i_cam, 1, 2] = A_entries[i_cam, 3]
            A[i_cam, 2, 2] = 1.0
    else:
        for i_cam in range(nCameras):
            A[i_cam, 0, 0] = A_entries[i_cam, 0, 0]
            A[i_cam, 0, 2] = A_entries[i_cam, 0, 2]
            A[i_cam, 1, 1] = A_entries[i_cam, 1, 1]
            A[i_cam, 1, 2] = A_entries[i_cam, 1, 2]
            A[i_cam, 2, 2] = A_entries[i_cam, 2, 2]
        
    # model
    joint_order = args['model']['joint_order'] # list
    joint_marker_order = args['model']['joint_marker_order'] # list
    skeleton_edges = args['model']['skeleton_edges'].cpu().numpy()
    skeleton_vertices = args['model']['skeleton_vertices'].cpu().numpy()
    skeleton_coords = args['model']['skeleton_coords'].cpu().numpy()
    skeleton_coords0 = args['model']['skeleton_coords0'].cpu().numpy()
    skeleton_vertices_links = args['model']['skeleton_vertices_links'].cpu().numpy()
    joint_marker_vec = args['model']['joint_marker_vectors'].cpu().numpy()
    joint_marker_index = args['model']['joint_marker_index'].cpu().numpy()
    #
    is_euler = args['model']['is_euler'].cpu().numpy()

    # initialize t and head direction
    labels3d = dict()
    for marker_index in range(nMarkers):
        marker_name = joint_marker_order[marker_index]
        string_split = marker_name.split('_')
        string = 'spot_' + '_'.join(string_split[1:-1])
        if ((string == 'spot_head_002') or
            (string == 'spot_head_003')):
            labels_use = labels[:, marker_index]
            if (np.sum(labels_use[:, 2] != 0.0) >= 2): # check if label was detected in at least two cameras
                labels3d[string] = interp_3d.calc_3d_point(labels_use, A, k, rX1, tX1)
    if (('spot_head_003' in labels3d) and ('spot_head_002' in labels3d)):
        head_direc = labels3d['spot_head_002'] - labels3d['spot_head_003']
        model_t = np.copy(labels3d['spot_head_003'])
    else: # use alternative initialization for t when the labels at the head are not visible
        avg_spine = np.zeros(3, dtype=np.float64)
        nSpine = 0
        avg_tail = np.zeros(3, dtype=np.float64)
        nTail = 0
        #
        calculate_3d_point = 0
        for marker_index in range(nMarkers):
            marker_name = joint_marker_order[marker_index]
            string_split = marker_name.split('_')
            if string_split[1] == 'spine':
                calculate_3d_point = 1
            elif (string_split[1] == 'tail'):
                calculate_3d_point = 2
            else:
                calculate_3d_point = 0
            if ((calculate_3d_point == 1) or (calculate_3d_point == 2)):
                string = 'spot_' + '_'.join(string_split[1:-1])
                labels_use = labels[:, marker_index]
                if (np.sum(labels_use[:, 2] != 0.0) >= 2): # check if label was detected in at least two cameras
                    labels3d[string] = interp_3d.calc_3d_point(labels_use, A, k, rX1, tX1)
                    if (calculate_3d_point == 1):
                        avg_spine += labels3d[string]
                        nSpine += 1
                    elif (calculate_3d_point == 2):
                        avg_tail += labels3d[string]
                        nTail += 1
        avg_spine /= nSpine
        avg_tail /= nTail
        #
        head_direc = avg_tail - avg_spine
        if ('spot_head_003' in labels3d):
            model_t = np.copy(labels3d['spot_head_003'])
        else:
            model_t = avg_spine - 0.5 * head_direc
    
    # r
    model_r = np.zeros((nBones, 3), dtype=np.float64)
    current_coords = np.tile(np.identity(3, dtype=np.float64), (nBones, 1, 1))
    skeleton_coords_use = np.copy(skeleton_coords0)
    coord_emb = np.array([[0.0, 0.0, -1.0],
                          [-1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0]],
                         dtype=np.float64)
    # head direction of the model after aligned to arena coordinate system
    current_head_direc = coord_emb[:, 2] # i.e. np.array([-1.0, 0.0, 0.0], dtype=np.float64)
    current_head_direc_xy = current_head_direc[:2] / np.sqrt(np.sum(current_head_direc[:2]**2))
    # head direction calculated from labels
    head_direc_xy = head_direc[:2] / np.sqrt(np.sum(head_direc[:2]**2))
    # signed angle between the current and target head direction in the xy-plane
    ang1 = np.arctan2(current_head_direc_xy[1], current_head_direc_xy[0])
    ang2 = np.arctan2(head_direc_xy[1], head_direc_xy[0])
    ang_xy = ang2 - ang1
    # resulting rotation matrix
    R_xy = rout_m.rodrigues2rotMat_single(np.array([0.0, 0.0, ang_xy], dtype=np.float64))
    #
    target_skeleton_coords = np.copy(skeleton_coords_use) # use this for mean pose (according to bounds)
    target_skeleton_coords = np.einsum('ij,njk->nik', coord_emb, target_skeleton_coords)
    target_skeleton_coords = np.einsum('ij,njk->nik', R_xy, target_skeleton_coords)
    
    # this is the first global rotation
    # it roughly aligns the head direction
    index_bone = 0
    R = np.dot(target_skeleton_coords[index_bone], current_coords[index_bone].T)
    model_r[index_bone] = rout_m.rotMat2rodrigues_single(R)
    current_coords = np.einsum('nij,jk->nik', current_coords, R)
        
    # mean values for angles (calculated from joint angle limits)
    bounds = args['bounds_pose'].cpu().numpy()
    for index_bone in range(1, nBones):
        joint_index = skeleton_edges[index_bone, 0]
        joint_name = joint_order[joint_index]
        bounds_use = bounds[3*(1+index_bone):3*(1+index_bone)+3]
        index_set_zero1 = np.array([np.all(bounds_use[0] == 0.0),
                                    np.all(bounds_use[1] == 0.0),
                                    np.all(bounds_use[2] == 0.0)], dtype=bool)
        index_set_zero2 = np.array([np.all(np.isinf(bounds_use[0])),
                                    np.all(np.isinf(bounds_use[1])),
                                    np.all(np.isinf(bounds_use[2]))], dtype=bool)
        index_set_zero = np.logical_or(index_set_zero1, index_set_zero2)
        model_r[index_bone][~index_set_zero] = np.mean(bounds_use[~index_set_zero], 1)
        model_r[index_bone][index_set_zero] = 0.0
    
    # to avoid rodrigues vectors with zero elements
    rodrigues_mask = ~is_euler[1:] & np.all(model_r == 0.0, 1)
    noise = np.random.randn(*np.shape(model_r[rodrigues_mask])) * 2**-23
    model_r[rodrigues_mask] += noise
    
    # construct initial x
    x_ini = np.concatenate([model_t.ravel(),
                            model_r.ravel()], 0)
    return x_ini


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
    nFree_bones = args['nFree_bones']
    nFree_markers = args['nFree_markers']
    nFree_pose = args['nFree_pose']

    print(cfg.file_labelsManual)
    # load frame list according to manual labels
    if (cfg.file_labelsManual[-3:] == 'npz'):
        labels_manual = np.load(cfg.file_labelsManual, allow_pickle=True)['arr_0'].item()
    elif (cfg.file_labelsManual[-3:] == 'npy'):
        labels_manual = np.load(cfg.file_labelsManual, allow_pickle=True).item()
    frame_list_manual = sorted(list(labels_manual.keys()))
    
    # get calibration frame list
    print(type(cfg.index_frames_calib))
    print(cfg.index_frames_calib)
    if isinstance(cfg.index_frames_calib,str):
        print('a')
        frame_list_calib = list(labels_manual.keys())
    else:
        print('b')
        frame_list_calib = np.array([], dtype=np.int64)
        for i in range(np.size(cfg.index_frames_calib, 0)):
            framesList_single = np.arange(cfg.index_frames_calib[i][0],
                                        cfg.index_frames_calib[i][1] + cfg.dFrames_calib,
                                        cfg.dFrames_calib,
                                        dtype=np.int64)
            frame_list_calib = np.concatenate([frame_list_calib, framesList_single], 0)
            
    nFrames = int(np.size(frame_list_calib))

    # create correct free_para
    free_para = np.concatenate([free_para_bones,
                                free_para_markers], 0)
    for i_frame in frame_list_calib:
        free_para = np.concatenate([free_para,
                                    free_para_pose], 0)
        
    # initialize x_pose
    # load arena coordinate system
    origin, coord = get_origin_coord(cfg.file_origin_coord, cfg.scale_factor)
    #
    labels_frame = np.zeros((nCameras, nMarkers, 3), dtype=np.float64)
    labels_use = labels_manual[frame_list_calib[0]]
    for i_marker in range(nMarkers):
        marker_name = joint_marker_order[i_marker]
        marker_name_split = marker_name.split('_')
        label_name = 'spot_' + '_'.join(marker_name_split[1:-1])
        if label_name in labels_use:
            labels_frame[:, i_marker, :2] = labels_use[label_name]
            labels_frame[:, i_marker, 2] = 1.0
    x_pose = initialize_x(args,
                          labels_frame,
                          coord, origin)[None, :]
    for i_frame in frame_list_calib[1:]:
        labels_frame = np.zeros((nCameras, nMarkers, 3), dtype=np.float64)
        labels_use = labels_manual[frame_list_calib[0]]
        for i_marker in range(nMarkers):
            marker_name = joint_marker_order[i_marker]
            marker_name_split = marker_name.split('_')
            label_name = 'spot_' + '_'.join(marker_name_split[1:-1])
            if label_name in labels_use:
                labels_frame[:, i_marker, :2] = labels_use[label_name]
                labels_frame[:, i_marker, 2] = 1.0
        x_pose_single = initialize_x(args,
                                     labels_frame,
                                     coord, origin)[None, :]
        x_pose = np.concatenate([x_pose, x_pose_single], 0)
    x_free_pose = x_pose[:, free_para_pose].ravel()
    x_pose = x_pose.ravel()
    
    # BOUNDS
    # bone_lengths
    bounds_free_bones = args['bounds_free_bones']
    bounds_free_low_bones = model.do_normalization_bones(bounds_free_bones[:, 0])
    bounds_free_high_bones = model.do_normalization_bones(bounds_free_bones[:, 1])
    # joint_marker_vec
    bounds_free_markers = args['bounds_free_markers']
    bounds_free_low_markers = model.do_normalization_markers(bounds_free_markers[:, 0])
    bounds_free_high_markers = model.do_normalization_markers(bounds_free_markers[:, 1])
    # pose
    bounds_free_pose = args['bounds_free_pose']
    bounds_free_low_pose_single = model.do_normalization(bounds_free_pose[:, 0][None, :], args).numpy().ravel()
    bounds_free_high_pose_single = model.do_normalization(bounds_free_pose[:, 1][None, :], args).numpy().ravel()
    bounds_free_low_pose = np.tile(bounds_free_low_pose_single, nFrames)
    bounds_free_high_pose = np.tile(bounds_free_high_pose_single, nFrames)
    # all
    bounds_free_low = np.concatenate([bounds_free_low_bones,
                                      bounds_free_low_markers,
                                      bounds_free_low_pose], 0)
    bounds_free_high = np.concatenate([bounds_free_high_bones,
                                       bounds_free_high_markers,
                                       bounds_free_high_pose], 0)
    bounds_free = np.stack([bounds_free_low, bounds_free_high], 1)
    args['bounds_free'] = bounds_free
    
    # INITIALIZE X
    inital_bone_lenght = 0.0
    inital_marker_length = 0.0
    # initialize bone_lengths and joint_marker_vec
    x_bones = bounds_free_low_bones.numpy() + (bounds_free_high_bones.numpy() - bounds_free_low_bones.numpy()) * 0.5
    x_bones[np.isinf(x_bones)] = inital_bone_lenght   

    x_free_bones = x_bones[free_para_bones]
    x_markers = np.full(nPara_markers, 0.0, dtype=np.float64)
    x_free_markers = np.zeros(nFree_markers ,dtype=np.float64)
    x_free_markers[(bounds_free_low_markers != 0.0) & (bounds_free_high_markers == 0.0)] = -inital_marker_length
    x_free_markers[(bounds_free_low_markers == 0.0) & (bounds_free_high_markers != 0.0)] = inital_marker_length
    x_free_markers[(bounds_free_low_markers != 0.0) & (bounds_free_high_markers != 0.0)] = 0.0
    x_free_markers[(bounds_free_low_markers == 0.0) & (bounds_free_high_markers == 0.0)] = 0.0
    x_markers[free_para_markers] = np.copy(x_free_markers)
    #
    x = np.concatenate([x_bones,
                        x_markers,
                        x_pose], 0)
    
    # update args regarding fixed tensors
    args['plot'] = False
    args['nFrames'] = nFrames
    
    # update args regarding x0 and labels
    # ARGS X
    args['x_torch'] = torch.from_numpy(np.concatenate([x_bones,
                                                       x_markers,
                                                       x_pose[:nPara_pose]], 0))
    args['x_free_torch'] = torch.from_numpy(np.concatenate([x_free_bones,
                                                            x_free_markers,
                                                            x_free_pose], 0))
    args['x_free_torch'].requires_grad = True
    # ARGS LABELS MANUAL
    args['labels_single_torch'] = torch.zeros((nFrames, nCameras, nMarkers, 3), dtype=model.float_type)
    args['labels_mask_single_torch'] = torch.zeros((nFrames, nCameras, nMarkers), dtype=torch.bool)
    for i in range(nFrames):
        index_frame = frame_list_calib[i]
        if index_frame in labels_manual:
            labels_manual_frame = labels_manual[index_frame]
            for marker_index in range(nMarkers):
                marker_name = joint_marker_order[marker_index]
                string = 'spot_' + '_'.join(marker_name.split('_')[1:-1])
                if string in labels_manual_frame:
                    mask = ~np.any(np.isnan(labels_manual[index_frame][string]), 1)
                    args['labels_mask_single_torch'][i, :, marker_index] = torch.from_numpy(mask)
                    args['labels_single_torch'][i, :, marker_index, :2][mask] = torch.from_numpy(labels_manual[index_frame][string][mask])
                    args['labels_single_torch'][i, :, marker_index, 2][mask] = 1.0
    
    # print ratio
    print('Ratio:')
    print('Number of free parameters:\t{:06d}'.format(int(np.sum(free_para))))
    print('Number of measurement:\t\t{:06d}'.format(int(2 * torch.sum(args['labels_mask_single_torch']))))
    

    # normalize x
    x_free = np.concatenate([model.do_normalization_bones(torch.from_numpy(x_free_bones[None, :])).numpy().ravel(),
                             model.do_normalization_markers(torch.from_numpy(x_free_markers[None, :])).numpy().ravel(),
                             model.do_normalization(torch.from_numpy(x_free_pose.reshape(nFrames, nFree_pose)), args).numpy().ravel()], 0)
    
    # OPTIMIZE
    # create optimization dictonary
    opt_options = dict()
    opt_options['disp'] = cfg.opt_options_calib__disp
    opt_options['maxiter'] = cfg.opt_options_calib__maxiter
    opt_options['maxcor'] = cfg.opt_options_calib__maxcor
    opt_options['ftol'] = cfg.opt_options_calib__ftol
    opt_options['gtol'] = cfg.opt_options_calib__gtol
    opt_options['maxfun'] = cfg.opt_options_calib__maxfun
    opt_options['iprint'] = cfg.opt_options_calib__iprint
    opt_options['maxls'] = cfg.opt_options_calib__maxls
    opt_dict = dict()
    opt_dict['opt_method'] = cfg.opt_method_calib
    opt_dict['opt_options'] = opt_options  
    print('Calibrating')
    min_result = opt.optimize__scipy(x_free, args,
                                     opt_dict)
    # copy fitting result into correct arrary
    x_fit_free = np.copy(min_result.x)
    print('Finished calibrating')
    print()

    # reverse normalization of x
    x_fit_free = np.concatenate([model.undo_normalization_bones(torch.from_numpy(x_fit_free[:nFree_bones].reshape(1, nFree_bones))).numpy().ravel(),
                                 model.undo_normalization_markers(torch.from_numpy(x_fit_free[nFree_bones:nFree_bones+nFree_markers].reshape(1, nFree_markers))).numpy().ravel(),
                                 model.undo_normalization(torch.from_numpy(x_fit_free[nFree_bones+nFree_markers:]).reshape(nFrames, nFree_pose), args).numpy().ravel()], 0)
    
    # add free variables
    x_calib = np.copy(x)
    x_calib[free_para] = x_fit_free
    
    # save
    np.save(cfg.folder_calib + '/x_calib.npy', x_calib)

if __name__ == "__main__":
    main()
