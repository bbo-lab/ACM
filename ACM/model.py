import math
import torch

import configuration as cfg

float_type = torch.float64
num_tol = cfg.num_tol

custom_clipping_constant = (0.25 * math.pi)**0.5 * cfg.slope

def do_custom_clip(z_in):
    return torch.erf(custom_clipping_constant * z_in) # [-1.0, 1.0]

def undo_custom_clip(z_in):
    return torch.erfinv(z_in) / custom_clipping_constant # [-inf, inf]

def do_normalization_r(r, args):
    return undo_custom_clip((r - args['bounds_free_pose_0'][None, 6:]) / args['bounds_free_pose_range'][None, 6:])

def undo_normalization_r(r_norm, args):
    return do_custom_clip(r_norm) * args['bounds_free_pose_range'][None, 6:] + args['bounds_free_pose_0'][None, 6:]

def do_normalization_r0(r0):
    return r0 / cfg.normalize_r0

def undo_normalization_r0(r0_norm):
    return r0_norm * cfg.normalize_r0

def do_normalization_t0(t0):
    return t0 / cfg.normalize_t0

def undo_normalization_t0(t0_norm):
    return t0_norm * cfg.normalize_t0

def do_normalization(x_free, args_torch):
    if (args_torch['use_custom_clip']):
        x_free_norm = torch.cat([do_normalization_t0(x_free[:, :3]),
                                 do_normalization_r0(x_free[:, 3:6]),
                                 do_normalization_r(x_free[:, 6:], args_torch)], 1)
    else:
        x_free_norm = torch.cat([do_normalization_t0(x_free[:, :3]),
                                 do_normalization_r0(x_free[:, 3:])], 1)
    return x_free_norm

def undo_normalization(x_free_norm, args_torch):
    if (args_torch['use_custom_clip']):
        x_free = torch.cat([undo_normalization_t0(x_free_norm[:, :3]),
                            undo_normalization_r0(x_free_norm[:, 3:6]),
                            undo_normalization_r(x_free_norm[:, 6:], args_torch)], 1)
    else:
        x_free = torch.cat([undo_normalization_t0(x_free_norm[:, :3]),
                            undo_normalization_r0(x_free_norm[:, 3:])], 1)
    return x_free

def do_normalization_bones(bone_lengths):
    return bone_lengths / cfg.normalize_bone_lengths

def undo_normalization_bones(bone_lengths_norm):
    return bone_lengths_norm * cfg.normalize_bone_lengths

def do_normalization_markers(joint_marker_vec):
    return joint_marker_vec / cfg.normalize_joint_marker_vec

def undo_normalization_markers(joint_marker_vec_norm):
    return joint_marker_vec_norm * cfg.normalize_joint_marker_vec

# implementation according to: 2009__Moll__Ball_Joints_for_Marker-less_Human_Motion_Capture
def rodrigues2rotMat(r): # nSigmaPoints, 3
    sqrt_arg = torch.sum(r**2, 1)
    theta = torch.sqrt(sqrt_arg)
    omega = r / theta[:, None]
    zero_entries = torch.zeros_like(theta, dtype=float_type)
    omega_hat = torch.stack([torch.stack([zero_entries, -omega[:, 2], omega[:, 1]], 1),
                             torch.stack([omega[:, 2], zero_entries, -omega[:, 0]], 1),
                             torch.stack([-omega[:, 1], omega[:, 0], zero_entries], 1)], 1)

    rotMat = torch.diag_embed(torch.ones_like(r, dtype=float_type)) + \
             torch.sin(theta)[:, None, None] * omega_hat + \
             (1.0 - torch.cos(theta))[:, None, None] * torch.einsum('nij,njk->nik', (omega_hat, omega_hat))
    
    mask = torch.all(abs(r) <= num_tol, 1)
    if torch.any(mask):
        rotMat[mask] = torch.diag_embed(torch.ones_like(r[mask], dtype=float_type))
    return rotMat # nSigmaPoints, 3, 3

def map_m(RX1, tX1, A, k,
          M): # nSigmaPoints, nMarkers, 3
    # RX1 * m + tX1
    m_cam = torch.einsum('cij,mnj->mcni', (RX1, M)) + tX1[None, :, None, :] # nSigmaPoints, nCameras, nMarkers, 3
    # m / m[2]
    m = m_cam[:, :, :, :2] / m_cam[:, :, :, 2][:, :, :, None] # nSigmaPoints, nCameras, nMarkers, 2
    # distort & A * m
    r_2 = m[:, :, :, 0]**2 + m[:, :, :, 1]**2
    sum_term = 1.0 + \
               k[None, :, None, 0] * r_2 + \
               k[None, :, None, 1] * r_2**2 + \
               k[None, :, None, 4] * r_2**3
    x_times_y_times_2 = m[:, :, :, 0] * m[:, :, :, 1] * 2.0    
    m = torch.stack([A[None, :, None, 1] + A[None, :, None, 0] * \
                     (m[:, :, :, 0] * sum_term + \
                      k[None, :, None, 2] * x_times_y_times_2 + \
                      k[None, :, None, 3] * (r_2 + 2.0 * m[:, :, :, 0]**2)),
                     A[None, :, None, 3] + A[None, :, None, 2] * \
                     (m[:, :, :, 1] * sum_term + \
                      k[None, :, None, 2] * (r_2 + 2.0 * m[:, :, :, 1]**2) + \
                      k[None, :, None, 3] * x_times_y_times_2)], 3)
    return m # nSigmaPoints, nCameras, nMarkers, 2

# IMPROVEMENT: bone_lengths & joint_marker_vec should have the correct size (i.e. < nBones/nMarkers due to symmetry)
def adjust_joint_marker_pos2(model_torch,
                             bone_lengths, joint_marker_vec,
                             model_t0_torch, model_r0_torch, model_r_torch,
                             nBones,
                             adjust_surface=False):
    skel_verts = model_torch['skeleton_vertices']
    skel_edges = model_torch['skeleton_edges']
    skel_verts_links = model_torch['skeleton_vertices_links']
    skel_coords_index = model_torch['skeleton_coords_index']
    joint_marker_index = model_torch['joint_marker_index']
#     skel_coords = model_torch['skeleton_coords']
    skel_coords0 = model_torch['skeleton_coords0']
    bone_lengths_index = model_torch['bone_lengths_index']
    joint_marker_vec_index = model_torch['joint_marker_vec_index']
    I_bone = model_torch['I_bone']
    is_euler = model_torch['is_euler'][2:] # [0] is global translation, [1] is global rotation
    #
    nSigmaPoints = model_t0_torch.size()[0]
    skel_verts_new = model_torch['skeleton_vertices_new'].repeat(nSigmaPoints, 1, 1)
    joint_marker_vec_new = model_torch['joint_marker_vectors_new'].repeat(nSigmaPoints, 1, 1)
    
    bone_lengths_sym = bone_lengths[:, bone_lengths_index]
    joint_marker_vec_sym = joint_marker_vec[:, abs(joint_marker_vec_index) - 1]
    mask_marker_sym = (joint_marker_vec_index < 0)
    joint_marker_vec_sym[:, mask_marker_sym, 0] = -joint_marker_vec_sym[:, mask_marker_sym, 0]
    
    # BONE COORDINATE SYSTEMS
    # always use rodrigues parameterization for first rotation to avoid gimbal lock
    R_T = rodrigues2rotMat(model_r0_torch).transpose(1, 2)
    skel_coords_new = torch.einsum('sij,bjk->sbik', (R_T, I_bone))
    # the for loop goes through the skeleton in a directed order
    # rotates all skeleton coordinate systems that are affected by the rotation of bone i_bone
    for i_bone in range(nBones-1):
        if (is_euler[i_bone]):
            cos_x = torch.cos(model_r_torch[:, i_bone, 0])
            sin_x = torch.sin(model_r_torch[:, i_bone, 0])
            cos_y = torch.cos(model_r_torch[:, i_bone, 1])
            sin_y = torch.sin(model_r_torch[:, i_bone, 1])
            cos_z = torch.cos(model_r_torch[:, i_bone, 2])
            sin_z = torch.sin(model_r_torch[:, i_bone, 2])
            R_T = torch.stack([torch.stack([cos_y*cos_z,
                                            sin_x*sin_y*cos_z + cos_x*sin_z,
                                            -cos_x*sin_y*cos_z + sin_x*sin_z], 1),
                               torch.stack([-cos_y*sin_z,
                                            -sin_x*sin_y*sin_z + cos_x*cos_z,
                                            cos_x*sin_y*sin_z + sin_x*cos_z], 1),
                               torch.stack([sin_y,
                                            -sin_x*cos_y,
                                            cos_x*cos_y], 1)], 1)
        else:
            R_T = rodrigues2rotMat(model_r_torch[:, i_bone]).transpose(1, 2)
        skel_coords_new[:, skel_verts_links[i_bone+1]] = torch.einsum('sij,sbjk->sbik', (R_T, skel_coords_new[:, skel_verts_links[i_bone+1]]))     
    skel_coords_new = skel_coords_new.transpose(2, 3) # to apply rotations starting from leaf joints
    skel_coords_new = torch.einsum('sbij,bjk->sbik', (skel_coords_new, skel_coords0))

    # SKELETON & MARKER
    # this moves all the other markers and the skeleton
    for i_bone in range(nBones):
        index_bone_start = skel_edges[i_bone, 0]
        index_bone_end = skel_edges[i_bone, 1]
        # SKELETON
        skel_verts_new[:, index_bone_end] = skel_verts_new[:, index_bone_start] + (skel_coords_new[:, i_bone, :, 2] * bone_lengths_sym[:, i_bone][:, None])
        # MARKER
        mask_markers = (joint_marker_index == index_bone_end)
        if torch.any(mask_markers):
            joint_marker_vec_new[:, mask_markers] = torch.einsum('sij,sbj->sbi', (skel_coords_new[:, i_bone], joint_marker_vec_sym[:, mask_markers])) 
    # translation: the position of first vertex in the skeleton graph becomes equal to the translation vector (first vertex position is always (0, 0, 0))
    skel_verts_new = skel_verts_new + model_t0_torch[:, None, :]
    
    # add joint position to get final marker positions
    marker_pos_new = skel_verts_new[:, joint_marker_index] + joint_marker_vec_new
    
    # SURFACE (optional, not used during optimization)
    if (adjust_surface): # should raise an error for nSigmaPoints != 1 (i.e. does not work then)
#         if ('surface_vertices' in model_torch.keys()):
#             surf_verts = model_torch['surface_vertices']
#             surf_verts_weights = model_torch['surface_vertices_weights']
#             surf_verts_new = torch.zeros_like(surf_verts, dtype=float_type)
#             for i_bone in range(nBones):
#                 index_bone_start = skel_edges[i_bone, 0]
#                 index_bone_end = skel_edges[i_bone, 1]
#                 skel_coords_new_use = skel_coords_new[0, i_bone]
#                 mask_surf = (surf_verts_weights[:, index_bone_end] != 0.0)
#                 skin_pos_norm = surf_verts[mask_surf] - skel_verts[index_bone_start]
#                 skin_pos_new = torch.einsum('ij,bj->bi', (skel_coords_new_use, skin_pos_norm)) + skel_verts_new[0, index_bone_start]
#                 surf_verts_new[mask_surf] = surf_verts_new[mask_surf] + surf_verts_weights[mask_surf, index_bone_end][:, None] * skin_pos_new
#         else:
#             skel_coords_new = float('nan')
#             surf_verts_new = float('nan')
        surf_verts_new = float('nan')
        return skel_coords_new, skel_verts_new, surf_verts_new, marker_pos_new
    else:
        return marker_pos_new # nSigmaPoints, nMarkers, xy
    
def fcn_emission(x_torch, args_torch):
    nPara_skel = args_torch['nPara_bones'] + args_torch['nPara_markers']
        
    nSigmaPoints = x_torch.size()[0]
    model_bone_lengths = x_torch[:, :args_torch['nPara_bones']].reshape(nSigmaPoints, args_torch['nPara_bones'])
    joint_marker_vec = x_torch[:, args_torch['nPara_bones']:nPara_skel].reshape(nSigmaPoints,
                                                                                args_torch['numbers']['nMarkers'], 3)
    model_t0_torch = x_torch[:, nPara_skel:nPara_skel+3].reshape(nSigmaPoints, 3)
    model_r0_torch = x_torch[:, nPara_skel+3:nPara_skel+6].reshape(nSigmaPoints, 3)
    model_r_torch = x_torch[:, nPara_skel+6:].reshape(nSigmaPoints, args_torch['numbers']['nBones']-1, 3)    
    
    if (args_torch['plot']):
        _, skel_pos_torch, _, marker_pos_torch = \
            adjust_joint_marker_pos2(args_torch['model'],
                                     model_bone_lengths, joint_marker_vec,
                                     model_t0_torch, model_r0_torch, model_r_torch,
                                     args_torch['numbers']['nBones'],
                                     True)
    else:
        marker_pos_torch = adjust_joint_marker_pos2(args_torch['model'],
                                                    model_bone_lengths, joint_marker_vec,
                                                    model_t0_torch, model_r0_torch, model_r_torch,
                                                    args_torch['numbers']['nBones'])
    marker_proj_torch = map_m(args_torch['calibration']['RX1_fit'],
                              args_torch['calibration']['tX1_fit'],
                              args_torch['calibration']['A_fit'],
                              args_torch['calibration']['k_fit'],
                              marker_pos_torch).reshape(nSigmaPoints,
                                                        args_torch['numbers']['nCameras']*args_torch['numbers']['nMarkers']*2)
    
    if (args_torch['plot']):
        return marker_proj_torch, marker_pos_torch, skel_pos_torch
    else:
        return marker_proj_torch

def fcn_emission_free(x_free_torch, args_torch):
    free_para_bones = args_torch['free_para_bones']
    free_para_markers = args_torch['free_para_markers']
    free_para_pose = args_torch['free_para_pose']
    nPara_bones = args_torch['nPara_bones']
    nPara_markers = args_torch['nPara_markers']
    nFree_bones = args_torch['nFree_bones']
    nFree_markers = args_torch['nFree_markers']
    
    nSigmaPoints = x_free_torch.size()[0]
    x_torch = args_torch['x_torch'].repeat(nSigmaPoints, 1)

    nPara_skel = nPara_bones + nPara_markers
    nFree_skel = nFree_bones + nFree_markers
    
    x_bones_torch = x_torch[:, :nPara_bones]
    x_bones_torch[:, free_para_bones] = undo_normalization_bones(x_free_torch[:, :nFree_bones])
    x_markers_torch = x_torch[:, nPara_bones:nPara_skel]
    x_markers_torch[:, free_para_markers] = undo_normalization_markers(x_free_torch[:, nFree_bones:nFree_skel])
    x_pose_torch = x_torch[:, nPara_skel:]
    x_pose_torch[:, free_para_pose] = undo_normalization(x_free_torch[:, nFree_skel:], args_torch)
    x_torch = torch.cat([x_bones_torch,
                         x_markers_torch,
                         x_pose_torch], 1)
    
    if (args_torch['plot']):
        marker_proj_torch, marker_pos_torch, skel_pos_torch = fcn_emission(x_torch, args_torch)
        marker_proj_torch[:, 0::2] = marker_proj_torch[:, 0::2] / (cfg.normalize_camera_sensor_x * 0.5) - 1.0
        marker_proj_torch[:, 1::2] = marker_proj_torch[:, 1::2] / (cfg.normalize_camera_sensor_y * 0.5) - 1.0
        return marker_proj_torch, marker_pos_torch, skel_pos_torch
    else:
        marker_proj_torch = fcn_emission(x_torch, args_torch)
        marker_proj_torch[:, 0::2] = marker_proj_torch[:, 0::2] / (cfg.normalize_camera_sensor_x * 0.5) - 1.0
        marker_proj_torch[:, 1::2] = marker_proj_torch[:, 1::2] / (cfg.normalize_camera_sensor_y * 0.5) - 1.0
        return marker_proj_torch

# WARNING: ONLY USE THIS WHEN MOVEMENT MODEL IS FIXED TO THE IDENTITY (I.E. A = I)!
def fcn_transition_free(z_tm1, M_transition, args_torch):
    return z_tm1

def obj_fcn(x_free_torch, args_torch):
    nFrames = args_torch['nFrames']
    nFree_bones = args_torch['nFree_bones']
    nFree_markers = args_torch['nFree_markers']
    nFree_pose = args_torch['nFree_pose']
    
    x_free_skel_torch = x_free_torch[:nFree_bones+nFree_markers].repeat(nFrames, 1)
    x_free_pose_torch = x_free_torch[nFree_bones+nFree_markers:].reshape(nFrames, nFree_pose)
    x_free_use_torch = torch.cat([x_free_skel_torch, x_free_pose_torch], 1)
    marker_proj_torch = fcn_emission_free(x_free_use_torch, args_torch).reshape(nFrames,
                                                                                args_torch['numbers']['nCameras'],
                                                                                args_torch['numbers']['nMarkers'],
                                                                                2)
    # reverse normalization
    diff_x_torch = ((marker_proj_torch[:, :, :, 0] + 1.0) * (cfg.normalize_camera_sensor_x * 0.5) - \
                    args_torch['labels_single_torch'][:, :, :, 0])
    diff_y_torch = ((marker_proj_torch[:, :, :, 1] + 1.0) * (cfg.normalize_camera_sensor_y * 0.5) - \
                    args_torch['labels_single_torch'][:, :, :, 1])
    
    # reduce influence of markers connected to the same joint via weights
    dist_torch = ((args_torch['weights'][None, None, :] * diff_x_torch)**2 + \
                  (args_torch['weights'][None, None, :] * diff_y_torch)**2)
    dist_torch[~args_torch['labels_mask_single_torch']] = 0.0 # set distances of undetected labels to zero

    # normalize by the number of used labels (i.e. get avg. distance per frame per camera per label per xy-position)
    res_torch = torch.sum(dist_torch) / torch.sum(args_torch['labels_mask_single_torch'], dtype=float_type) 
    return res_torch