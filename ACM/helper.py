#!/usr/bin/env python3

import copy
import numpy as np
import os
import sys
import torch
import warnings

import configuration as cfg

from . import anatomy
from . import routines_math as rout_m


def get_calibration(file_origin_coord, file_calibration,
                    scale_factor):
    calibration = np.load(file_calibration, allow_pickle=True).item()

    # This is mostly for backwards compatibility. Newer calibrations from calibcam 1.2.0 onwards
    # should always have a scale factor of 1.
    if 'scale_factor' in calibration:
        if scale_factor is None:
            scale_factor = calibration['scale_factor']
        elif not scale_factor == calibration['scale_factor']:
            warnings.warn("!!!!!!! scale_factor in calibration does not match scale_factor in configuration !!!!!!!")

    n_cameras = calibration['nCameras']
    A = calibration['A_fit']
    k = calibration['k_fit']
    rX1 = calibration['rX1_fit']
    RX1 = calibration['RX1_fit']
    tX1 = calibration['tX1_fit']  
    
    if os.path.isfile(file_origin_coord):
        origin_coord_arena = np.load(file_origin_coord, allow_pickle=True).item()
        origin_arena = origin_coord_arena['origin']
        coord_arena = origin_coord_arena['coord']
    else:
        origin_arena = np.zeros(3, dtype=np.float64)
        coord_arena = np.identity(3, dtype=np.float64)
        print('WARNING: File that defines the origin does not exist')
    
    # change reference coordinate system to coordinate system of arena
    tX1 = tX1 + np.einsum('nij,j->ni', RX1, origin_arena)
    RX1 = np.einsum('nij,jk->nik', RX1, coord_arena)
    for i_cam in range(n_cameras):
        rX1[i_cam] = rout_m.rotMat2rodrigues_single(RX1[i_cam])
        
    # scaling (calibration board square size -> cm)
    tX1 = tX1 * scale_factor

    calibration_torch = dict()
#     calibration_torch['nCameras'] = int(nCameras)
    if len(A.shape) == 2:  # Old style calibration
        calibration_torch['A_fit'] = torch.from_numpy(A)
    else:
        calibration_torch['A_fit'] = torch.from_numpy(np.array([
            A[:, 0, 0],
            A[:, 0, 2],
            A[:, 1, 1],
            A[:, 1, 2],
        ]).T)
    calibration_torch['k_fit'] = torch.from_numpy(k)
    calibration_torch['rX1_fit'] = torch.from_numpy(rX1)
    calibration_torch['RX1_fit'] = torch.from_numpy(RX1)
    calibration_torch['tX1_fit'] = torch.from_numpy(tX1)
    calibration_torch['scale_factor'] = scale_factor
    return calibration_torch


def get_model3d(file_model):
    # load
    model3d = np.load(file_model, allow_pickle=True).item()
    joint_order = model3d['joint_order']
    joint_marker_order = model3d['joint_marker_order']
    
    skeleton_edges = model3d['skeleton_edges']
    skeleton_vertices = model3d['skeleton_vertices']
    skeleton_coords = model3d['skeleton_coords']
    joint_order = model3d['joint_order']
    
    skel_verts_new = np.zeros_like(model3d['skeleton_vertices'], dtype=np.float64)[None, :, :]
    joint_marker_vec_new = np.zeros_like(model3d['joint_marker_vectors'], dtype=np.float64)[None, :, :]
    
    
    nBones = np.shape(skeleton_edges)[0]
    nMarkers = np.shape(joint_marker_order)[0]
        
    joint_marker_vec_index = anatomy.get_joint_marker_vec_index(nMarkers, joint_marker_order)
    bone_lengths_index = anatomy.get_bone_lengths_index(nBones, skeleton_edges, joint_order)
    bounds, _, _, is_euler = anatomy.get_bounds_pose(nBones, skeleton_edges, joint_order)
    I_bone = np.tile(np.identity(3, dtype=np.float64), (nBones, 1, 1))

    skeleton_coords0 = anatomy.get_skeleton_coords0(skeleton_edges, skeleton_vertices, skeleton_coords, joint_order, bounds)
    
    # reparameterization
    if (cfg.use_reparameterization):
        skeleton_vertices_links = model3d['skeleton_vertices_links']
        skeleton_coords_index = model3d['skeleton_coords_index']
        skeleton_coords0 = anatomy.reparameterize_coords0(skeleton_coords0, bounds,
                                                          skeleton_vertices_links, skeleton_coords_index, is_euler)

    model3d_torch = dict()
    #
    model3d_torch['skeleton_vertices'] = torch.from_numpy(model3d['skeleton_vertices'])
    model3d_torch['skeleton_edges'] = torch.from_numpy(model3d['skeleton_edges'])
    model3d_torch['bone_lengths'] = torch.from_numpy(model3d['bone_lengths'])
    model3d_torch['skeleton_vertices_links'] = torch.from_numpy(model3d['skeleton_vertices_links']) 
    model3d_torch['joint_marker_vectors'] = torch.from_numpy(model3d['joint_marker_vectors']) 
    model3d_torch['skeleton_coords_index'] = torch.from_numpy(model3d['skeleton_coords_index'])
    model3d_torch['joint_marker_index'] = torch.from_numpy(model3d['joint_marker_index'])
    model3d_torch['skeleton_coords'] = torch.from_numpy(model3d['skeleton_coords'])
    #
    model3d_torch['skeleton_coords0'] = torch.from_numpy(skeleton_coords0)
    model3d_torch['I_bone'] = torch.from_numpy(I_bone)
    model3d_torch['is_euler'] = torch.from_numpy(is_euler)
    model3d_torch['skeleton_vertices_new'] = torch.from_numpy(skel_verts_new)
    model3d_torch['joint_marker_vectors_new'] = torch.from_numpy(joint_marker_vec_new)
    # for plotting
    model3d_torch['surface_triangles'] = torch.from_numpy(model3d['surface_triangles'])
    model3d_torch['surface_vertices'] = torch.from_numpy(model3d['surface_vertices'])
    model3d_torch['surface_vertices_weights'] = torch.from_numpy(model3d['surface_vertices_weights'])
    # for calculating other things (i.e. nor needed within model.py)
    model3d_torch['joint_order'] = list(joint_order)
    model3d_torch['joint_marker_order'] = list(joint_marker_order)
    #
    model3d_torch['joint_marker_vec_index'] = torch.from_numpy(joint_marker_vec_index)
    model3d_torch['bone_lengths_index'] = torch.from_numpy(bone_lengths_index)
    return model3d_torch


def get_labelsDLC(file_labelsDLC, pcutoff,
                  joint_marker_order, nCameras, nMarkers):
    # load labels
    labelsDLC = np.load(file_labelsDLC, allow_pickle=True).item()
    nFrames = np.size(labelsDLC['frame_list'], 0)    
    labels_mask = (labelsDLC['labels_all'][:, :, :, 2] > pcutoff)
    labels = np.zeros((nFrames, nCameras, nMarkers, 3),
                       dtype=np.float64)
    labels[labels_mask] = labelsDLC['labels_all'][labels_mask] # shoule be possible to just copy all respective label values
    return labels, labels_mask


def get_arguments(file_origin_coord, file_calibration, file_model, file_labelsDLC,
                  scale_factor, pcutoff=0.9):
    # calibration
    calibration = get_calibration(file_origin_coord, file_calibration,
                                  scale_factor)
    nCameras = int(np.size(calibration['A_fit'], 0))
    # model3d
    model3d = get_model3d(file_model)
    joint_order = model3d['joint_order'] # list
    joint_marker_order = model3d['joint_marker_order'] # list
    skeleton_edges = model3d['skeleton_edges'].cpu().numpy()
    joint_marker_vectors = model3d['joint_marker_vectors'].cpu().numpy()
    joint_marker_index = model3d['joint_marker_index'].cpu().numpy()

    # numbers
    nBones = int(np.size(skeleton_edges, 0))
#     nJoints = int(np.size(model3d['skeleton_vertices'], 0))
    nMarkers = int(np.size(joint_marker_vectors, 0))
    numbers = dict()
    numbers['nCameras'] = int(nCameras)
    numbers['nBones'] = int(nBones)
#     numbers['nJoints'] = int(nJoints)
    numbers['nMarkers'] = int(nMarkers)
#     numbers['nLabels'] = int(nLabels)
#     numbers['nFrames'] = int(nFrames)
    
    # labels
    labels, labels_mask = \
        get_labelsDLC(file_labelsDLC, pcutoff,
                      joint_marker_order, nCameras, nMarkers)
#     nLabels = int(np.size(labelsDLC['labels_all'], 2))
#     nFrames = (np.size(labelsDLC['frame_list'], 0))
    


    # generate weights for objective function
    weights = np.ones(nMarkers, dtype=np.float64)
#     for i_joint in np.unique(joint_marker_index):
#         mask = (joint_marker_index == i_joint)
#         weights[mask] = 1.0 / np.sum(mask, dtype=np.float64)

    # bounds & free parameters
    bounds_bones, bounds_free_bones, free_para_bones = anatomy.get_bounds_bones(nBones, skeleton_edges, joint_order)
    bounds_markers, bounds_free_markers, free_para_markers = anatomy.get_bounds_markers(nMarkers, joint_marker_order)
    bounds_pose, bounds_free_pose, free_para_pose, is_euler = anatomy.get_bounds_pose(nBones, skeleton_edges, joint_order)
    # reparameterization
    if (cfg.use_reparameterization):
        bounds_pose, bounds_free_pose = anatomy.reparameterize_bounds(bounds_pose, nBones, free_para_pose)
    
    # mean/range of pose parameters
    bounds_free_pose_range = (bounds_free_pose[:, 1] - bounds_free_pose[:, 0]) / 2.0
    # to avoid warnings:
    bounds_free_pose_0 = np.zeros_like(bounds_free_pose[:, 0], dtype=np.float64)
    mask_inf = np.isinf(bounds_free_pose_range)
    bounds_free_pose_0[mask_inf] = 0.0
    bounds_free_pose_0[~mask_inf] = bounds_free_pose[~mask_inf, 0] + bounds_free_pose_range[~mask_inf]
    
    # initialize args
    args = dict()
    # numbers
    args['numbers'] = numbers
    # calibration
    args['calibration'] = calibration
    # model3d
    args['model'] = model3d
    # bounds
    args['free_para_bones'] = torch.from_numpy(free_para_bones)
    args['bounds_bones'] = torch.from_numpy(bounds_bones)
    args['bounds_free_bones'] = torch.from_numpy(bounds_free_bones)
    args['nPara_bones'] = int(np.size(free_para_bones))
    args['nFree_bones'] = int(np.sum(free_para_bones))
    #
    args['free_para_markers'] = torch.from_numpy(free_para_markers)
    args['bounds_markers'] = torch.from_numpy(bounds_markers)
    args['bounds_free_markers'] = torch.from_numpy(bounds_free_markers)
    args['nPara_markers'] = int(np.size(free_para_markers))
    args['nFree_markers'] = int(np.sum(free_para_markers))
    #
    args['free_para_pose'] = torch.from_numpy(free_para_pose)
    args['bounds_pose'] = torch.from_numpy(bounds_pose)
    args['nPara_pose'] = int(np.size(free_para_pose))
    args['nFree_pose'] = int(np.sum(free_para_pose))
    #
    args['bounds_free_pose'] = torch.from_numpy(bounds_free_pose)
    args['bounds_free_pose_range'] = torch.from_numpy(bounds_free_pose_range)
    args['bounds_free_pose_0'] = torch.from_numpy(bounds_free_pose_0)
    args['is_euler'] = torch.from_numpy(is_euler)
    # weights
    args['weights'] = torch.from_numpy(weights)
    # labels
    args['labels'] = torch.from_numpy(labels)
    args['labels_mask'] = torch.from_numpy(labels_mask)
#     # ATTENTION: labels_batch and labels_mask_batch need to be populated with their respective entries later on!
#     # ATTENTION: Do not initialize the arrays with np.nan since that can lead to errors in the gradient calculation in pytorch (even when using masks)
#     labels_batch = np.zeros((nBatch, nCameras, nMarkers, 3), dtype=np.float64)
#     labels_mask_batch = np.zeros((nBatch, nCameras, nMarkers), dtype=bool)
    return args

# put this into the functions in optimization.py
def update_args(x_torch, i_frame, args):
    free_para = args['free_para']

    args['x_torch'] = x_torch[None, :].clone()
#     args['x_torch'].requires_grad = True
    args['x_free_torch'] = x_torch[free_para][None, :].clone()
    args['x_free_torch'].requires_grad = True
    args['labels_single_torch'] = args['labels'][i_frame].clone()
    args['labels_mask_single_torch'] = args['labels_mask'][i_frame].clone()
    return
