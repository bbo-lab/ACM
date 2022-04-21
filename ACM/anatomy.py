#!/usr/bin/env python3

import numpy as np

import configuration as cfg

from . import helper
from . import routines_math as rout_m


def get_index_bone_start(str, skeleton_edges, joint_order):
    nBones = np.size(skeleton_edges, 0)
    index_bone = np.nan
    for i_joint in range(nBones + 1):
        if (joint_order[i_joint] == str):
            index_bone_set = np.where(skeleton_edges[:, 0] == i_joint)[0]
            if (np.size(index_bone_set) > 0):
                index_bone = index_bone_set[0]
            break
    return index_bone

def get_index_bone_end(str, skeleton_edges, joint_order):
    nBones = np.size(skeleton_edges, 0)
    index_bone = np.nan
    for i_joint in range(nBones + 1):
        if (joint_order[i_joint] == str):
            index_bone_set = np.where(skeleton_edges[:, 1] == i_joint)[0]
            if (np.size(index_bone_set) > 0):
                index_bone = index_bone_set[0]
            break
    return index_bone

def get_joint_marker_vec_index(nMarkers, joint_marker_order):
    joint_marker_vec_index = np.ones(nMarkers, dtype=np.int64)
    joint_marker_vec_index_entry = 0
    for i_marker in range(nMarkers):
        marker_name = joint_marker_order[i_marker]
        marker_name_split = marker_name.split('_')
        mask = np.array([i == 'right' for i in marker_name_split], dtype=bool)
        marker_is_symmetric = np.any(mask)
        if (marker_is_symmetric):
            index_left_right = np.arange(len(mask), dtype=np.int64)[mask][0]
            index1 = joint_marker_order.index('_'.join(marker_name_split[:index_left_right]) + 
                                              '_left_' + 
                                              '_'.join(marker_name_split[index_left_right+1:]))
            joint_marker_vec_index[i_marker] = -joint_marker_vec_index[index1]
        else:
            joint_marker_vec_index[i_marker] = joint_marker_vec_index_entry + 1
            joint_marker_vec_index_entry += 1
    return joint_marker_vec_index

def get_bone_lengths_index(nBones, skeleton_edges, joint_order):
    bone_lengths_index = np.ones(nBones, dtype=np.int64)
    bone_lengths_index_entry = 0
    for i_bone in range(nBones):
        joint_name = joint_order[skeleton_edges[i_bone, 1]]
        joint_name_split = joint_name.split('_')
        mask = np.array([i == 'right' for i in joint_name_split], dtype=bool)
        joint_is_symmetric = np.any(mask)
        if (joint_is_symmetric):            
            index_left_right = np.arange(len(mask), dtype=np.int64)[mask][0]
            index1 = joint_order.index('_'.join(joint_name_split[:index_left_right]) + 
                                       '_left' + '_' * len(joint_name_split[index_left_right+1:]) + 
                                       '_'.join(joint_name_split[index_left_right+1:]))
            index2 = np.where(skeleton_edges[:, 1] == index1)[0][0]
            bone_lengths_index[i_bone] = bone_lengths_index[index2]
        else:
            bone_lengths_index[i_bone] = bone_lengths_index_entry
            bone_lengths_index_entry += 1          
    return bone_lengths_index

def get_bounds_bones(nBones, skeleton_edges, joint_order):    
#     # 2018__Prodinger__Whole_bone_testing_in_small_animals:_systematic_characterization_of_the_mechanical_properties_of_different_rodent_bones_available_for_rat_fracture_models
#     mean_femur = 3.89 # cm
#     std_femur = 0.406386515524322 # cm
#     mean_tibia = 4.18 # cm
#     std_tibia = 0.4069397989875161 # cm
#     mean_humerus = 2.9875 # cm
#     std_humerus = 0.3287761396451999 # cm

    if not hasattr(cfg,'species'): 
        cfg.species = 'rat'
        print()
    
    if cfg.species=='rat':
        # 2002__Lammers__Ontogenetic_allometry_in_the_locomotor_skeleton_of_specialize_half-bounding_mammals
        mean_humerus = 0.0075 * float(cfg.body_weight) # cm
        std_humerus = 0.0005 * float(cfg.body_weight) # cm
        mean_radius = 0.0069 * float(cfg.body_weight) # cm
        std_radius = 0.0004 * float(cfg.body_weight) # cm
        mean_metacarpal = 0.0023 * float(cfg.body_weight) # cm
        std_metacarpal = 0.0001 * float(cfg.body_weight) # cm
        mean_femur = 0.0102 * float(cfg.body_weight) # cm
        std_femur = 0.0006 * float(cfg.body_weight) # cm
        mean_tibia = 0.0114 * float(cfg.body_weight) # cm
        std_tibia = 0.0006 * float(cfg.body_weight) # cm
        mean_metatarsal = 0.0053 * float(cfg.body_weight) # cm
        std_metatarsal = 0.0003 * float(cfg.body_weight) # cm
    elif cfg.species=='mouse':
        # Marchini M, Silva Hernandez E, Rolian C. Morphology and development of a novel murine skeletal dysplasia. PeerJ. 2019;7:e7180. Published 2019 Jul 4. doi:10.7717/peerj.7180
        # adult control lengths calculated by DW in Dropbox (NIG)\8_rat full body tracking\Re_submission January 2022\MouseLimbBoneLengthData\peerj-07-7180-s007.xlsx
        mean_humerus = 1.162 # cm
        std_humerus = 0.039 # cm
        mean_radius = 1.531 # cm
        std_radius = 0.046 # cm
        mean_metacarpal = 0.192 # cm
        std_metacarpal = 0.009 # cm
        mean_femur = 1.668 # cm
        std_femur = 0.029 # cm
        mean_tibia = 1.809 # cm
        std_tibia = 0.052 # cm
        mean_metatarsal = 0.639 # cm
        std_metatarsal = 0.020 # cm
    else:
        print('ERROR: Unknown species.')
        exit()
    
    bounds_low = np.full(nBones, 0.0, dtype=np.float64)
    bounds_high = np.full(nBones, np.inf, dtype=np.float64)
    
    bone_lengths_index = get_bone_lengths_index(nBones, skeleton_edges, joint_order)

    # FRONT LIMBS
    # Clavicle (spine - shoulder)
    joint_name = 'joint_shoulder_left'
    index_bone = get_index_bone_end(joint_name, skeleton_edges, joint_order)
    bounds_low[bone_lengths_index[index_bone]] = 0.0
    bounds_high[bone_lengths_index[index_bone]] = np.inf
    # Humerus (shoulder - elbow)
    joint_name = 'joint_elbow_left'
    index_bone = get_index_bone_end(joint_name, skeleton_edges, joint_order)
    bounds_low[bone_lengths_index[index_bone]] = max(0.0, mean_humerus - std_humerus * float(cfg.sigma_factor))
    bounds_high[bone_lengths_index[index_bone]] = mean_humerus + std_humerus * float(cfg.sigma_factor)
    # Radius / Ulna (elbow - wrist)
    joint_name = 'joint_wrist_left'
    index_bone = get_index_bone_end(joint_name, skeleton_edges, joint_order)
    bounds_low[bone_lengths_index[index_bone]] = max(0.0, mean_radius - std_radius * float(cfg.sigma_factor))
    bounds_high[bone_lengths_index[index_bone]] = mean_radius + std_radius * float(cfg.sigma_factor)
#     # Metacarpal (wrist - ) # not part of the model since wrist is directly connected to a finger
    # HIND LIMBS
    # Pelvis (spine - hip)
    joint_name = 'joint_hip_left'
    index_bone = get_index_bone_end(joint_name, skeleton_edges, joint_order)
    bounds_low[bone_lengths_index[index_bone]] = 0.0
    bounds_high[bone_lengths_index[index_bone]] = np.inf
    # Femur (hip - knee)
    joint_name = 'joint_knee_left'
    index_bone = get_index_bone_end(joint_name, skeleton_edges, joint_order)
    bounds_low[bone_lengths_index[index_bone]] = max(0.0, mean_femur - std_femur * float(cfg.sigma_factor))
    bounds_high[bone_lengths_index[index_bone]] = mean_femur + std_femur * float(cfg.sigma_factor)
    # Tibia / Fibula (knee - ankle)
    joint_name = 'joint_ankle_left'
    index_bone = get_index_bone_end(joint_name, skeleton_edges, joint_order)
    bounds_low[bone_lengths_index[index_bone]] = max(0.0, mean_tibia - std_tibia * float(cfg.sigma_factor))
    bounds_high[bone_lengths_index[index_bone]] = mean_tibia + std_tibia * float(cfg.sigma_factor)
    # Metatarsal (ankle - paw)
    joint_name = 'joint_paw_hind_left'
    index_bone = get_index_bone_end(joint_name, skeleton_edges, joint_order)
    bounds_low[bone_lengths_index[index_bone]] = max(0.0, mean_metatarsal - std_metatarsal * float(cfg.sigma_factor))
    bounds_high[bone_lengths_index[index_bone]] = mean_metatarsal + std_metatarsal * float(cfg.sigma_factor)

    # make arrays 1-dimensional
    free_para = (bounds_low != bounds_high)
    free_para = free_para.ravel()
    bounds_low = bounds_low.ravel()
    bounds_high = bounds_high.ravel()
    
    # generate bounds and bounds_free
    bounds = np.stack([bounds_low, bounds_high], 1)
    bounds_free = np.stack([bounds_low[free_para],
                            bounds_high[free_para]], 1)
    return bounds, bounds_free, free_para

def get_bounds_markers(nMarkers, joint_marker_order):
    bounds_low = np.full((nMarkers, 3), -np.inf, dtype=np.float64)
    bounds_high = np.full((nMarkers, 3), np.inf, dtype=np.float64)

    joint_marker_vec_index = get_joint_marker_vec_index(nMarkers, joint_marker_order)
    joint_marker_vec_index_use = abs(joint_marker_vec_index) - 1
    
    # x: left/right
    # y: up/down
    # z: front/back
    # LIMBS
    # front
    marker_name = 'marker_shoulder_left_start'
    if marker_name in joint_marker_order:
        marker_index = joint_marker_order.index(marker_name)
        bounds_low[joint_marker_vec_index_use[marker_index], 0] = 0.0 # marker is after joint
        bounds_high[joint_marker_vec_index_use[marker_index], 0] = 0.0 # marker is not allowed to move in plane
        bounds_low[joint_marker_vec_index_use[marker_index], 1] = 0.0
        bounds_low[joint_marker_vec_index_use[marker_index], 2] = 0.0 # marker is on the left
        bounds_high[joint_marker_vec_index_use[marker_index], 2] = 0.0 # marker is not allowed to move in plane
    marker_name = 'marker_elbow_left_start'
    if marker_name in joint_marker_order:
        marker_index = joint_marker_order.index(marker_name)
        bounds_high[joint_marker_vec_index_use[marker_index], 0] = 0.0
        bounds_low[joint_marker_vec_index_use[marker_index], 1] = 0.0
        bounds_high[joint_marker_vec_index_use[marker_index], 1] = 0.0
        bounds_low[joint_marker_vec_index_use[marker_index], 2] = 0.0
        bounds_high[joint_marker_vec_index_use[marker_index], 2] = 0.0
    marker_name = 'marker_paw_front_left_start' # i.e. wrist
    if marker_name in joint_marker_order:
        marker_index = joint_marker_order.index(marker_name)
        bounds_low[joint_marker_vec_index_use[marker_index], 0] = 0.0
        bounds_high[joint_marker_vec_index_use[marker_index], 0] = 0.0    
        bounds_high[joint_marker_vec_index_use[marker_index], 1] = 0.0
        bounds_low[joint_marker_vec_index_use[marker_index], 2] = 0.0
        bounds_high[joint_marker_vec_index_use[marker_index], 2] = 0.0
    for marker_name in list(['marker_finger_left_001_start',
                             'marker_finger_left_002_start',
                             'marker_finger_left_003_start']):
        if marker_name in joint_marker_order:
            marker_index = joint_marker_order.index(marker_name)
            bounds_low[joint_marker_vec_index_use[marker_index], 1] = 0.0
            bounds_high[joint_marker_vec_index_use[marker_index], 1] = 0.0
    # hind
    marker_name = 'marker_hip_left_start'
    if marker_name in joint_marker_order:
        marker_index = joint_marker_order.index(marker_name)
        bounds_low[joint_marker_vec_index_use[marker_index], 0] = 0.0 # marker is not allowed to move in plane
        bounds_high[joint_marker_vec_index_use[marker_index], 0] = 0.0 # marker is before joint
        bounds_low[joint_marker_vec_index_use[marker_index], 1] = 0.0
        bounds_low[joint_marker_vec_index_use[marker_index], 2] = 0.0 # marker is on the left
    marker_name = 'marker_knee_left_start'
    if marker_name in joint_marker_order:
        marker_index = joint_marker_order.index(marker_name)
        bounds_high[joint_marker_vec_index_use[marker_index], 0] = 0.0
        bounds_low[joint_marker_vec_index_use[marker_index], 1] = 0.0
        bounds_high[joint_marker_vec_index_use[marker_index], 1] = 0.0
        bounds_low[joint_marker_vec_index_use[marker_index], 2] = 0.0
        bounds_high[joint_marker_vec_index_use[marker_index], 2] = 0.0 
    marker_name = 'marker_ankle_left_start'
    if marker_name in joint_marker_order:
        marker_index = joint_marker_order.index(marker_name)
        bounds_high[joint_marker_vec_index_use[marker_index], 0] = 0.0
        bounds_low[joint_marker_vec_index_use[marker_index], 1] = 0.0
        bounds_high[joint_marker_vec_index_use[marker_index], 1] = 0.0
        bounds_low[joint_marker_vec_index_use[marker_index], 2] = 0.0
        bounds_high[joint_marker_vec_index_use[marker_index], 2] = 0.0
    marker_name = 'marker_paw_hind_left_start'
    if marker_name in joint_marker_order:
        marker_index = joint_marker_order.index(marker_name)
        bounds_low[joint_marker_vec_index_use[marker_index], 0] = 0.0
        bounds_high[joint_marker_vec_index_use[marker_index], 0] = 0.0
        bounds_high[joint_marker_vec_index_use[marker_index], 1] = 0.0
        bounds_low[joint_marker_vec_index_use[marker_index], 2] = 0.0
        bounds_high[joint_marker_vec_index_use[marker_index], 2] = 0.0
    for marker_name in list(['marker_toe_left_001_start',
                             'marker_toe_left_002_start',
                             'marker_toe_left_003_start']):
        if marker_name in joint_marker_order:
            marker_index = joint_marker_order.index(marker_name)
            bounds_low[joint_marker_vec_index_use[marker_index], 1] = 0.0
            bounds_high[joint_marker_vec_index_use[marker_index], 1] = 0.0
    
    # SIDE
    marker_name = 'marker_side_left_start'
    if marker_name in joint_marker_order:
        marker_index = joint_marker_order.index(marker_name)
        bounds_high[joint_marker_vec_index_use[marker_index], 0] = 0.0 # marker is on the left
    
    # HEAD
    for marker_name in list(['marker_head_001_start',
                             'marker_head_002_start',
                             'marker_head_003_start']):
        if marker_name in joint_marker_order:
            marker_index = joint_marker_order.index(marker_name)
            bounds_low[joint_marker_vec_index_use[marker_index], 0] = 0.0
            bounds_high[joint_marker_vec_index_use[marker_index], 0] = 0.0
            bounds_low[joint_marker_vec_index_use[marker_index], 1] = 0.0

    # SPINE
    marker_name = 'marker_spine_006_start'
    if marker_name in joint_marker_order:
        marker_index = joint_marker_order.index(marker_name)        
        bounds_low[joint_marker_vec_index_use[marker_index], 0] = 0.0
        bounds_high[joint_marker_vec_index_use[marker_index], 0] = 0.0
        bounds_low[joint_marker_vec_index_use[marker_index], 1] = 0.0
        bounds_low[joint_marker_vec_index_use[marker_index], 2] = 0.0
        bounds_high[joint_marker_vec_index_use[marker_index], 2] = 0.0
    for marker_name in list(['marker_spine_001_start',
                             'marker_spine_002_start',
                             'marker_spine_003_start',
                             'marker_spine_004_start',
                             'marker_spine_005_start',
                             'marker_spine_006_start']):
        if marker_name in joint_marker_order:
            marker_index = joint_marker_order.index(marker_name)        
            bounds_low[joint_marker_vec_index_use[marker_index], 0] = 0.0
            bounds_high[joint_marker_vec_index_use[marker_index], 0] = 0.0
            bounds_low[joint_marker_vec_index_use[marker_index], 1] = 0.0
    # TAIL
    for marker_name in list(['marker_tail_001_start',
                             'marker_tail_002_start',
                             'marker_tail_003_start',
                             'marker_tail_004_start',
                             'marker_tail_005_start',
                             'marker_tail_006_start']):
        if marker_name in joint_marker_order:
            marker_index = joint_marker_order.index(marker_name)
            bounds_low[joint_marker_vec_index_use[marker_index], 0] = 0.0
            bounds_high[joint_marker_vec_index_use[marker_index], 0] = 0.0
            bounds_low[joint_marker_vec_index_use[marker_index], 1] = 0.0
    
    # markers to fix    
    fixed_marker_list = list(['marker_finger_left_002_start',
                              'marker_head_003_start',
                              'marker_tail_001_start',
                              'marker_toe_left_002_start']) # only fix leaf joints
    for marker_name in fixed_marker_list:
        if marker_name in joint_marker_order:
            marker_index = joint_marker_order.index(marker_name)
            bounds_low[joint_marker_vec_index_use[marker_index], :] = 0.0
            bounds_high[joint_marker_vec_index_use[marker_index], :] = 0.0
    
    # make arrays 1-dimensional
    free_para = (bounds_low != bounds_high)
    free_para = free_para.ravel()
    bounds_low = bounds_low.ravel()
    bounds_high = bounds_high.ravel()
    
    # generate bounds and bounds_free
    bounds = np.stack([bounds_low,
                       bounds_high], 1)
    bounds_free = np.stack([bounds_low[free_para],
                            bounds_high[free_para]], 1)
    
    return bounds, bounds_free, free_para

def get_bounds_pose(nBones, skeleton_edges, joint_order):
    # axis definitions (see get_skeleton_coord0)
    # x: flexion / extension
    # y: abbduction / adduction
    # z: rotation
    
    # define if parameters are euler angles or rodrigues vector
    is_euler = np.zeros(1+nBones, dtype=bool)
    
    # inital bounds for all bones
    bounds_low = np.zeros((1+nBones, 3), dtype=np.float64)
    bounds_high = np.zeros((1+nBones, 3), dtype=np.float64)
    
    # bounds for bones at head/spine/tail
    bounds_low[:] = np.array([-90.0, -90.0, 0.0], dtype=np.float64) * np.pi/180.0
    bounds_high[:] = np.array([90.0, 90.0, 0.0], dtype=np.float64) * np.pi/180.0
    
    # translation
    bounds_low[0] = np.full(3, -np.inf, dtype=np.float64)
    bounds_high[0] = np.full(3, np.inf, dtype=np.float64)
    
    # first global rotation (always uses Rodrigues parameterization)
    bounds_low[1] = np.full(3, -np.inf, dtype=np.float64)
    bounds_high[1] = np.full(3, np.inf, dtype=np.float64)

    # use joint angle limits of cat
    # use mean of range limits
    #
    # left - right:
    # x is the same
    # y is inverted
    # z is inverted
    #
    # LEFT:
    # SHOULDER:
    # x:
    # ini at 90 flexion
    # extension: +
    # flexion: -
    # [-65, 115] +90 -> [25, 205]
    # y:
    # ini at 0
    # adduction: +
    # abduction: -
    # [-85, 25]
    # z:
    # ini at 0
    # external: +
    # internal: -
    # [-35, 35]
    # ELBOW
    # x:
    # ini at 90 extension
    # flexion: +
    # extension: -
    # [-87.5, 55] +90 -> [2.5, 145]
    # y:
    # none
    # z:
    # ini at 0
    # pronation: +
    # supination: -
    # [-100, 45]
    # WRIST
    # x:
    # ini at 0
    # extension: +
    # flexion: -
    # [-135, 35]
    # y:
    # ini at 0
    # radial deviation: +
    # ulnar deviation: -
    # [-12.5, 37.5]
    # z:
    # none
    # HIP:
    # x:
    # ini at 90 flexion
    # extension: +
    # flexion: -
    # [-55, 105] +90 -> [35, 195]
    # y:
    # ini at 0
    # adduction: +
    # abduction: -
    # [-65, 25]
    # z:
    # ini at 0
    # internal rotation: +
    # external rotation: -
    # [-85, 40]
    # KNEE
    # x:
    # ini at 90 extension
    # extension: +
    # flexion: -
    # [-55, 105] -90 -> [-145, 15]
    # y:
    # none
    # z:
    # none
    # ANKLE
    # x:
    # ini at 90 extension
    # flexion: +
    # extension: -
    # [-100, 55] +90 -> [-10, 145]
    # y:
    # none
    # z:
    # none
    # HIND PAW (METATARSAL - TARSAL)
    # x:
    # none
    # y:
    # none
    # z:
    # ini at 0
    # eversion: +
    # inversion: -
    # [-15, 35]
    
    # FRONT LIMBS
    # shoulder
    joint_name = 'joint_shoulder_left'
    index_bone = get_index_bone_start(joint_name, skeleton_edges, joint_order)
    is_euler[1 + index_bone] = True
    bounds_low[1 + index_bone] = np.array([25, -85, -35], dtype=np.float64) * np.pi/180.0
    bounds_high[1 + index_bone] = np.array([205, 25, 35], dtype=np.float64) * np.pi/180.0
    joint_name = 'joint_shoulder_right'
    index_bone = get_index_bone_start(joint_name, skeleton_edges, joint_order)
    is_euler[1 + index_bone] = True
    bounds_low[1 + index_bone] = np.array([25, -25, -35], dtype=np.float64) * np.pi/180.0
    bounds_high[1 + index_bone] = np.array([205, 85, 35], dtype=np.float64) * np.pi/180.0
    # elbow
    joint_name = 'joint_elbow_left'
    index_bone = get_index_bone_start(joint_name, skeleton_edges, joint_order)
    is_euler[1 + index_bone] = True
    bounds_low[1 + index_bone] = np.array([2.5, 0, -100], dtype=np.float64) * np.pi/180.0
    bounds_high[1 + index_bone] = np.array([145, 0, 45], dtype=np.float64) * np.pi/180.0
    joint_name = 'joint_elbow_right'
    index_bone = get_index_bone_start(joint_name, skeleton_edges, joint_order)
    is_euler[1 + index_bone] = True
    bounds_low[1 + index_bone] = np.array([2.5, 0, -45], dtype=np.float64) * np.pi/180.0
    bounds_high[1 + index_bone] = np.array([145, 0, 100], dtype=np.float64) * np.pi/180.0
    # wrist
    joint_name = 'joint_wrist_left'
    index_bone = get_index_bone_start(joint_name, skeleton_edges, joint_order)
    is_euler[1 + index_bone] = True
    bounds_low[1 + index_bone] = np.array([-135, -12.5, 0], dtype=np.float64) * np.pi/180.0
    bounds_high[1 + index_bone] = np.array([35, 37.5, 0], dtype=np.float64) * np.pi /180.0
    joint_name = 'joint_wrist_right'
    index_bone = get_index_bone_start(joint_name, skeleton_edges, joint_order)
    is_euler[1 + index_bone] = True
    bounds_low[1 + index_bone] = np.array([-135, -37.5, 0], dtype=np.float64) * np.pi/180.0
    bounds_high[1 + index_bone] = np.array([35, 12.5, 0], dtype=np.float64) * np.pi/180.0
    # HIND LIMBS
    # hip
    joint_name = 'joint_hip_left'
    index_bone = get_index_bone_start(joint_name, skeleton_edges, joint_order)
    is_euler[1 + index_bone] = True
    bounds_low[1 + index_bone] = np.array([35, -65, -85], dtype=np.float64) * np.pi/180.0
    bounds_high[1 + index_bone] = np.array([195, 25, 40], dtype=np.float64) * np.pi/180.0
    joint_name = 'joint_hip_right'
    index_bone = get_index_bone_start(joint_name, skeleton_edges, joint_order)
    is_euler[1 + index_bone] = True
    bounds_low[1 + index_bone] = np.array([35, -25, -40], dtype=np.float64) * np.pi/180.0
    bounds_high[1 + index_bone] = np.array([195, 65, 85], dtype=np.float64) * np.pi/180.0
    # knee
    joint_name = 'joint_knee_left'
    index_bone = get_index_bone_start(joint_name, skeleton_edges, joint_order)
    is_euler[1 + index_bone] = True
    bounds_low[1 + index_bone] = np.array([-145, 0, 0], dtype=np.float64) * np.pi/180.0
    bounds_high[1 + index_bone] = np.array([15, 0, 0], dtype=np.float64) * np.pi/180.0
    joint_name = 'joint_knee_right'
    index_bone = get_index_bone_start(joint_name, skeleton_edges, joint_order)
    is_euler[1 + index_bone] = True
    bounds_low[1 + index_bone] = np.array([-145, 0, 0], dtype=np.float64) * np.pi/180.0
    bounds_high[1 + index_bone] = np.array([15, 0, 0], dtype=np.float64) * np.pi/180.0
    # ankle
    joint_name = 'joint_ankle_left'
    index_bone = get_index_bone_start(joint_name, skeleton_edges, joint_order)
    is_euler[1 + index_bone] = True
    bounds_low[1 + index_bone] = np.array([-10, 0, 0], dtype=np.float64) * np.pi/180.0
    bounds_high[1 + index_bone] = np.array([145, 0, 0], dtype=np.float64) * np.pi/180.0
    joint_name = 'joint_ankle_right'
    index_bone = get_index_bone_start(joint_name, skeleton_edges, joint_order)
    is_euler[1 + index_bone] = True
    bounds_low[1 + index_bone] = np.array([-10, 0, 0], dtype=np.float64) * np.pi/180.0
    bounds_high[1 + index_bone] = np.array([145, 0, 0], dtype=np.float64) * np.pi/180.0
    # hind paw (metatarsal - tarsal)
    joint_name = 'joint_paw_hind_left'
    index_bone = get_index_bone_start(joint_name, skeleton_edges, joint_order)
    is_euler[1 + index_bone] = True
    bounds_low[1 + index_bone] = np.array([0, 0, -15], dtype=np.float64) * np.pi/180.0
    bounds_high[1 + index_bone] = np.array([0, 0, 35], dtype=np.float64) * np.pi/180.0
    joint_name = 'joint_paw_hind_right'
    index_bone = get_index_bone_start(joint_name, skeleton_edges, joint_order)
    is_euler[1 + index_bone] = True
    bounds_low[1 + index_bone] = np.array([0, 0, -35], dtype=np.float64) * np.pi/180.0
    bounds_high[1 + index_bone] = np.array([0, 0, 15], dtype=np.float64) * np.pi/180.0
    
    # fixes collarbone
    joint_name = 'joint_shoulder_left'
    index_bone = get_index_bone_end(joint_name, skeleton_edges, joint_order)
    is_euler[1 + index_bone] = True
    bounds_low[1 + index_bone] = 0.0
    bounds_high[1 + index_bone] = 0.0
    joint_name = 'joint_shoulder_right'
    index_bone = get_index_bone_end(joint_name, skeleton_edges, joint_order)
    is_euler[1 + index_bone] = True
    bounds_low[1 + index_bone] = 0.0
    bounds_high[1 + index_bone] = 0.0
    # fixes pelvis
    joint_name = 'joint_hip_left'
    index_bone = get_index_bone_end(joint_name, skeleton_edges, joint_order)
    is_euler[1 + index_bone] = True
    bounds_low[1 + index_bone] = 0.0
    bounds_high[1 + index_bone] = 0.0
    joint_name = 'joint_hip_right'
    index_bone = get_index_bone_end(joint_name, skeleton_edges, joint_order)
    is_euler[1 + index_bone] = True
    bounds_low[1 + index_bone] = 0.0
    bounds_high[1 + index_bone] = 0.0

    # turn joint angle limits off for mode 1 and 3
    if ((cfg.mode == 1) or (cfg.mode == 3)):
        joints_list = list(['joint_shoulder_left', 'joint_shoulder_right',
                            'joint_elbow_left', 'joint_elbow_right',
                            'joint_wrist_left', 'joint_wrist_right',
                            'joint_hip_left', 'joint_hip_right',
                            'joint_knee_left', 'joint_knee_right',
                            'joint_ankle_left', 'joint_ankle_right',
                            'joint_paw_hind_left', 'joint_paw_hind_right'])
        for joint_name in joints_list:
            index_bone = get_index_bone_start(joint_name, skeleton_edges, joint_order)
            free_para_single = (bounds_low[1 + index_bone] != bounds_high[1 + index_bone])
            bounds_low[1 + index_bone, free_para_single] = -np.pi
            bounds_high[1 + index_bone, free_para_single] = np.pi

    # to make all joint angles rodrigues vectors
    if (cfg.use_rodrigues):
        is_euler = np.zeros(1+nBones, dtype=bool)
    
    # generate free_para
    free_para = (bounds_low != bounds_high)
    
    # make arrays 1-dimensional
    free_para = free_para.ravel()
    bounds_low = bounds_low.ravel()
    bounds_high = bounds_high.ravel()

    # generate bounds and bounds_free
    bounds = np.stack([bounds_low, bounds_high], 1)
    bounds_free = np.stack([bounds_low[free_para], bounds_high[free_para]], 1)
    
    return bounds, bounds_free, free_para, is_euler

def r2R_xyz(r):
    ang_x = r[0]
    ang_y = r[1]
    ang_z = r[2]
    
    R_xyz = np.array([[[1.0, 0.0, 0.0],
                      [0.0, np.cos(ang_x), -np.sin(ang_x)],
                      [0.0, np.sin(ang_x), np.cos(ang_x)]],
                     [[np.cos(ang_y), 0.0, np.sin(ang_y)],
                      [0.0, 1.0, 0.0],
                      [-np.sin(ang_y), 0.0, np.cos(ang_y)]],
                     [[np.cos(ang_z), -np.sin(ang_z), 0.0],
                      [np.sin(ang_z), np.cos(ang_z), 0.0],
                      [0.0, 0.0, 1.0]]],
                     dtype=np.float64)
    # equal to the order used within the objective function: 1. Z, 2. Y, 3. X
    R = np.dot(R_xyz[0], np.dot(R_xyz[1], R_xyz[2]))
    return R

# get skeleton coordinate systems
def get_skeleton_coords(skeleton_edges, skeleton_vertices, joint_order):        
    nEdges = np.size(skeleton_edges, 0)
    skeleton_coords = np.zeros((nEdges, 3, 3), dtype=np.float64)
    
    limb_list = list(['shoulder', 'elbow', 'wrist',
                      'hip', 'knee', 'ankle'])
    for i_edge in range(nEdges):
        z_vec = skeleton_vertices[skeleton_edges[i_edge, 1]] - \
                skeleton_vertices[skeleton_edges[i_edge, 0]]
        # to get rid of numerical rounding errors when an entry of bone_pos is actually 0 (not sure if necessary)
        z_vec = np.float16(z_vec)
        z_vec = z_vec / np.sqrt(np.sum(z_vec**2))

        # the method fails if the z-coordinate of z-vec is negative
        # solution: rotate the vector into the positive half and then rotate back
        # (use x-direction for rotation since it is the last rotation used in the model, i.e. order is z, y, x)
        z_is_negative = False
        if (z_vec[2] < 0.0):
            z_is_negative = True
            r = np.pi * np.array([1.0, 0.0, 0.0], dtype=np.float64)
            R = rout_m.rodrigues2rotMat_single(r)
            z_vec = np.dot(R, z_vec)
        
        bone_direc0 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        rot_axis_rodrigues = np.cross(bone_direc0, z_vec)
        ang_rodrigues = np.arcsin(np.sqrt(np.sum(rot_axis_rodrigues**2)))
        rot_axis_rodrigues = rot_axis_rodrigues / np.sqrt(np.sum(rot_axis_rodrigues**2))
        R = rout_m.rodrigues2rotMat_single(ang_rodrigues * rot_axis_rodrigues)
        
        # option 1
        ang_y1 = np.arcsin(R[0, 2])
        ang_x1 = np.arctan2(-R[1, 2] / np.cos(ang_y1),
                            R[2, 2] / np.cos(ang_y1))
        ang_z1 = np.arctan2(-R[0, 1] / np.cos(ang_y1),
                            R[0, 0] / np.cos(ang_y1))
        if (z_is_negative):
            ang_x_options = np.array([ang_x1 + np.pi, ang_x1 - np.pi], dtype=np.float64)
            min_change_index = np.argmin(np.abs(ang_x_options))
            ang_x1 = ang_x_options[min_change_index]             
        # option 2
        ang_y2 = np.pi - ang_y1
        ang_x2 = np.arctan2(-R[1, 2] / np.cos(ang_y2),
                            R[2, 2] / np.cos(ang_y2))
        ang_z2 = np.arctan2(-R[0, 1] / np.cos(ang_y2),
                            R[0, 0] / np.cos(ang_y2))
        if (z_is_negative):
            ang_x_options = np.array([ang_x2 + np.pi, ang_x2 - np.pi], dtype=np.float64)
            min_change_index = np.argmin(np.abs(ang_x_options))
            ang_x2 = ang_x_options[min_change_index]           

        # to decide which coordinate system to use
        # use the one where the rotation angles are the smallest
        res1 = ang_x1**2 + ang_y1**2 + ang_z1**2
        res2 = ang_x2**2 + ang_y2**2 + ang_z2**2
        ang_options = np.array([res1, res2], dtype=np.float64)
        min_change_index = np.argmin(ang_options)
        if (min_change_index == 0):
            ang_x = ang_x1
            ang_y = ang_y1
            ang_z = ang_z1
        else:
            ang_x = ang_x2
            ang_y = ang_y2
            ang_z = ang_z2
        
        R_xyz = np.array([[[1.0, 0.0, 0.0],
                          [0.0, np.cos(ang_x), -np.sin(ang_x)],
                          [0.0, np.sin(ang_x), np.cos(ang_x)]],
                         [[np.cos(ang_y), 0.0, np.sin(ang_y)],
                          [0.0, 1.0, 0.0],
                          [-np.sin(ang_y), 0.0, np.cos(ang_y)]],
                         [[np.cos(ang_z), -np.sin(ang_z), 0.0],
                          [np.sin(ang_z), np.cos(ang_z), 0.0],
                          [0.0, 0.0, 1.0]]], dtype=np.float64)
        R = np.dot(R_xyz[0], np.dot(R_xyz[1], R_xyz[2]))
        skeleton_coords[i_edge] = R
        
        
        # if the joint belongs to the limbs assume the following
        # two bones form a plane for the repsecitve joint
        # treat the normal of this plane as the axis for flexion/extension axis, i.e. x axis
        joint_index = skeleton_edges[i_edge, 0]
        joint_name = joints_order[joint_index]
        joint_name_split = joint_name.split('_')
        if (joint_name_split[1] in limb_list):
            i_edge_previous = np.where(skeleton_edges[i_edge, 0] == skeleton_edges[:, 1])[0][0]
            x_vec_previous = skeleton_coords[i_edge_previous, :, 0] #R[:, 0]
            z_vec_previous = skeleton_coords[i_edge_previous, :, 2]
            z_vec = R[:, 2]

            x_vec_new1 = np.cross(z_vec, z_vec_previous)
            x_vec_new1 = x_vec_new1 / np.sqrt(np.sum(x_vec_new1**2))
            x_vec_new2 = -x_vec_new1

            dot_prod1 = np.dot(x_vec_previous, x_vec_new1)
            ang1 = np.arccos(dot_prod1)
            dot_prod2 = np.dot(x_vec_previous, x_vec_new2)
            ang2 = np.arccos(dot_prod2)

            ang_options = np.array([ang1, ang2], dtype=np.float64)
            index_option = np.argmin([np.abs(ang_options)])

            if (index_option == 0):
                R[:, 0] = x_vec_new1
            else:
                R[:, 0] = x_vec_new2
            y_vec_new = np.cross(z_vec, R[:, 0])
            y_vec_new = y_vec_new / np.sqrt(np.sum(y_vec_new**2))
            R[:, 1] = y_vec_new

        skeleton_coords[i_edge] = R
    return skeleton_coords

def recover_coord_xyz(current_coord, target_coord,
                      bounds_low, bounds_high):
    R = np.dot(current_coord.T, target_coord)
        
    # implementation according to:
    # https://en.wikipedia.org/wiki/Euler_angles#Extrinsic_rotations [Tait-Bryan angles (order: X1 Y2 Z3)]
    # http://www.gregslabaugh.net/publications/euler.pdf
    ang_y1 = np.arcsin(R[0, 2])
    ang_y2 = np.pi - ang_y1
    ang_x1 = np.arctan2(-R[1, 2] / np.cos(ang_y1), R[2, 2] / np.cos(ang_y1))
    ang_x2 = np.arctan2(-R[1, 2] / np.cos(ang_y2), R[2, 2] / np.cos(ang_y2))
    ang_z1 = np.arctan2(-R[0, 1] / np.cos(ang_y1), R[0, 0] / np.cos(ang_y1))
    ang_z2 = np.arctan2(-R[0, 1] / np.cos(ang_y2), R[0, 0] / np.cos(ang_y2))

    # to decide which coordinate system to use
    # use the one where the rotation angles are the smallest
    # trying to stay within joint angle limits here
    # might need to be fixed (use joint angle limits
    # i.e. if (any(model_r[index_bone_end] > upper) or any(model_r[index_bone_end] < lower))
#         res1 = ang_x1**2 + ang_y1**2 + ang_z1**2
#         res2 = ang_x2**2 + ang_y2**2 + ang_z2**2
#         ang_options = np.array([res1, res2], dtype=np.float64)
#         min_change_index = np.argmin(ang_options)
#         if (min_change_index == 0):
#             ang_x = ang_x1
#             ang_y = ang_y1
#             ang_z = ang_z1
#         else:
#             ang_x = ang_x2
#             ang_y = ang_y2
#             ang_z = ang_z2  
    option1 = np.array([ang_x1, ang_y1, ang_z1], dtype=np.float16) # use np.float16 to get rid of small values due to numerical errors
    option2 = np.array([ang_x2, ang_y2, ang_z2], dtype=np.float16) # use np.float16 to get rid of small values due to numerical errors
    valid_bounds_array1 = (option1 >= bounds_low) & (option1 <= bounds_high)
    valid_bounds_array1[bounds_low == 0.0] = True
    valid_bounds_array1[bounds_high == 0.0] = True
    valid_bounds1 = np.all(valid_bounds_array1)
    valid_bounds_array2 = (option2 >= bounds_low) & (option2 <= bounds_high)
    valid_bounds_array2[bounds_low == 0.0] = True
    valid_bounds_array2[bounds_high == 0.0] = True
    valid_bounds2 = np.all(valid_bounds_array2)
    valid_bounds = np.array([valid_bounds1, valid_bounds2], dtype=bool)
    if (np.all(valid_bounds) or np.all(~valid_bounds)):
#         cond = (np.max(np.abs(option1)) < np.max(np.abs(option2)))  
        res1 = np.sum(option1**2)
        res2 = np.sum(option2**2)
        cond = (res1 < res2)
        if (cond):
            r = option1
        else:
            r = option2
    else:
        if (valid_bounds1):
            r = option1
        else:
            r = option2
    return r

# ATTENTION: hard coded
# get roation values to compensate for rigid structure of pelvis and collarbone (i.e. different inital orientation)
def get_skeleton_coords0(skeleton_edges, skeleton_vertices, skeleton_coords, joint_order, bounds):    
    joint_name = 'joint_hip_left'
    index_bone_hip_left = get_index_bone_end(joint_name, skeleton_edges, joint_order)
    index_joint_hip_left = skeleton_edges[index_bone_hip_left, 1]
    joint_name = 'joint_hip_right'
    index_bone_hip_right = get_index_bone_end(joint_name, skeleton_edges, joint_order)
    index_joint_hip_right = skeleton_edges[index_bone_hip_right, 1]
    joint_name = 'joint_shoulder_left'
    index_bone_shoulder_left = get_index_bone_end(joint_name, skeleton_edges, joint_order)
    index_joint_shoulder_left = skeleton_edges[index_bone_shoulder_left, 1]
    joint_name = 'joint_shoulder_right'
    index_bone_shoulder_right = get_index_bone_end(joint_name, skeleton_edges, joint_order)
    index_joint_shoulder_right = skeleton_edges[index_bone_shoulder_right, 1]
        
    # vector form pelvis left to right (should be [1.0, 0.0] anyway)
    pelvis_vec = skeleton_vertices[index_joint_hip_right] - skeleton_vertices[index_joint_hip_left]
    pelvis_vec_xy = pelvis_vec[:-1]
    pelvis_vec_xy = pelvis_vec_xy / np.sqrt(np.sum(pelvis_vec_xy**2))
    # vector form collarbone left to right
    collarbone_vec = skeleton_vertices[index_joint_shoulder_right] - skeleton_vertices[index_joint_shoulder_left]
    collarbone_vec_xy = collarbone_vec[:-1]
    collarbone_vec_xy = collarbone_vec_xy / np.sqrt(np.sum(collarbone_vec_xy**2))
    
    # angle beteween pelvis and collarbone (assumes pelvis defines coord0)
    ang = np.arccos(np.dot(collarbone_vec_xy, pelvis_vec_xy))
    R = rout_m.rodrigues2rotMat_single(ang * np.array([0.0, 0.0, 1.0], dtype=np.float64))

    # assumes pelvis defines coord0
    coord0 = np.identity(3, dtype=np.float64)
    nBones = np.size(skeleton_edges, 0)
    skel_coords0 = np.tile(coord0, (nBones, 1, 1))
    
    # it is actually not necessary to use recover_coord_xyz here
    # could just "recover" the target coordinate system with the rodrigues parameterization
    bounds_low = np.zeros(3, dtype=np.float64) # do not use bounds since coord0 should be recovered
    bounds_high = np.zeros(3, dtype=np.float64) # do not use bounds since coord0 should be recovered
    # hip left
    coord = skeleton_coords[index_bone_hip_left]
    model_r0 = recover_coord_xyz(coord0, coord,
                                 bounds_low, bounds_high)
    skel_coords0[index_bone_hip_left] = r2R_xyz(model_r0)
    # hip right
    coord = skeleton_coords[index_bone_hip_right]
    model_r0 = recover_coord_xyz(coord0, coord,
                                 bounds_low, bounds_high)
    skel_coords0[index_bone_hip_right] = r2R_xyz(model_r0)
    # shoulder left
    coord = skeleton_coords[index_bone_shoulder_left]
    coord = np.dot(R, coord)
    model_r0 = recover_coord_xyz(coord0, coord,
                                 bounds_low, bounds_high)
    skel_coords0[index_bone_shoulder_left] = r2R_xyz(model_r0)
    # shoulder right
    coord = skeleton_coords[index_bone_shoulder_right]
    coord = np.dot(R, coord)
    model_r0 = recover_coord_xyz(coord0, coord,
                                 bounds_low, bounds_high)
    skel_coords0[index_bone_shoulder_right] = r2R_xyz(model_r0)
    
    return skel_coords0    

### to reparameterize so that inital skeleton coordinates match the mean of the bounds
def reparameterize_coords0(skel_coords0, bounds,
                           skel_verts_links, skel_coords_index, is_euler):
    nBones = np.size(skel_coords0, 0)
    coord0 = np.identity(3, dtype=np.float64)
    skel_coords_new = np.tile(coord0, (nBones, 1, 1))
    #
    bounds_low = np.reshape(bounds[:, 0], (nBones+1, 3))
    bounds_high = np.reshape(bounds[:, 1], (nBones+1, 3))
    bounds_range = bounds_high - bounds_low
    
    
    model_r = np.zeros((nBones+1, 3), dtype=np.float64)
    mask_inf = np.isinf(bounds_range)
    model_r[mask_inf] = 0.0
    model_r[~mask_inf] = bounds_low[~mask_inf] + bounds_range[~mask_inf]/2.0
    model_r = model_r[2:]
    for i_bone in range(nBones-1):
        if (is_euler[2:][i_bone]):
            cos_x = np.cos(model_r[i_bone, 0])
            sin_x = np.sin(model_r[i_bone, 0])
            cos_y = np.cos(model_r[i_bone, 1])
            sin_y = np.sin(model_r[i_bone, 1])
            cos_z = np.cos(model_r[i_bone, 2])
            sin_z = np.sin(model_r[i_bone, 2])
            R_x = np.array([[1.0, 0.0, 0.0],
                            [0.0, cos_x, -sin_x],
                            [0.0, sin_x, cos_x]], dtype=np.float64)
            R_y = np.array([[cos_y, 0.0, sin_y],
                            [0.0, 1.0, 0.0],
                            [-sin_y, 0.0, cos_y]], dtype=np.float64)
            R_z = np.array([[cos_z, -sin_z, 0.0],
                            [sin_z, cos_z, 0.0],
                            [0.0, 0.0, 1.0]], dtype=np.float64)
            R_xyz_T = np.dot(R_z, np.dot(R_y, R_x)).transpose(1, 0)
            
        else:
            if (np.all(abs(model_r[i_bone]) <= 2**-23)):
                rotMat = np.identity(3, dtype=np.float64)
            else:
                theta = np.sum(model_r[i_bone]**2)**0.5
                omega = model_r[i_bone] / theta
                omega_hat = np.array([[0.0, -omega[2], omega[1]],
                                      [omega[2], 0.0, -omega[0]],
                                      [-omega[1], omega[0], 0.0]], dtype=np.float64)
                rotMat = np.identity(3, dtype=np.float64) + \
                         np.sin(theta) * omega_hat + \
                         (1.0 - np.cos(theta)) * np.dot(omega_hat, omega_hat)
            R_xyz_T = rotMat.transpose(1, 0)
        mask = (skel_verts_links[i_bone+1] == True)
        skel_coords_new[mask] = np.einsum('ij,njk->nik', R_xyz_T, skel_coords_new[mask])          
    skel_coords_new = skel_coords_new.transpose(0, 2, 1)
    skel_coords_new = np.einsum('nij,njk->nik', skel_coords_new, skel_coords0)

    return skel_coords_new

### to reparameterize so that inital skeleton coordinates match the mean of the bounds
def reparameterize_bounds(bounds, nBones, free_para):
    bounds_low = np.reshape(bounds[:, 0], (nBones+1, 3))
    bounds_high = np.reshape(bounds[:, 1], (nBones+1, 3))
    bounds_range = bounds_high - bounds_low
    
    bounds_low_new = np.copy(bounds_low)
    bounds_high_new = np.copy(bounds_high)
    bounds_low_new[2:] = -bounds_range[2:]/2.0
    bounds_high_new[2:] = bounds_range[2:]/2.0
    bounds_new = np.stack([bounds_low_new.ravel(),
                           bounds_high_new.ravel()], 1)

    bounds_free_new = bounds_new[free_para]
    return bounds_new, bounds_free_new

# ATTENTION: hard coded
def get_coord0(skeleton_edges, skeleton_vertices, joint_order):
    # x: flexion / extension
    # y: abbduction / adduction
    # z: rotation
    
    joint_name1 = 'joint_hip_left'
    index1_bone = get_index_bone_end(joint_name1, skeleton_edges, joint_order)
    index1_joint = skeleton_edges[index1_bone, 1]
    index1_connecting_joint = skeleton_edges[index1_bone, 0]
    joint_name2 = 'joint_hip_right'
    index2_bone = get_index_bone_end(joint_name2, skeleton_edges, joint_order)
    index2_joint = skeleton_edges[index2_bone, 1]
    index2_connecting_joint = skeleton_edges[index2_bone, 0]
    
    if (index1_connecting_joint != index2_connecting_joint):
        raise AssertionError('ERROR: {:s} and {:s} are not connected to the same start joint'.format(joint_name1, joint_name2))
        
    vert0 = skeleton_vertices[index1_connecting_joint]
    vert1 = skeleton_vertices[index1_joint]
    vert2 = skeleton_vertices[index2_joint]
    
    joint1 = vert1 - vert0
    joint2 = vert2 - vert0
    # z (rotation)
    z_vec = joint1 + 0.5 * (joint2 - joint1)
    z_vec_norm = np.sqrt(np.sum(z_vec**2))
    z_vec = z_vec / z_vec_norm
    # y (abbduction / adduction)
    y_vec = np.cross(joint1, joint2)
    y_vec_norm = np.sqrt(np.sum(y_vec**2))
    y_vec = y_vec / y_vec_norm
    # x (flexion / extension)
    x_vec = np.cross(y_vec, z_vec)
    x_vec_norm = np.sqrt(np.sum(x_vec**2))
    x_vec = x_vec / x_vec_norm

    coord0 = np.zeros((3, 3), dtype=np.float64)
    coord0[:, 0] = np.copy(x_vec)
    coord0[:, 1] = np.copy(y_vec)
    coord0[:, 2] = np.copy(z_vec)
    return coord0
