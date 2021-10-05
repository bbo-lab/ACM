#!/usr/bin/env python3

import os
import numpy as np
import torch
from scipy.io import savemat

import configuration as cfg

from . import em
from . import helper
from . import kalman
from . import model

def gen_args_Qg(args):
    measure_mask = args['args_kalman']['measure_mask'].cpu().numpy().astype(bool)
    #
    nMeasureT = np.sum(measure_mask, 0).astype(np.float64)
    nMeasureT_times_log2pi = np.sum(nMeasureT * np.log(2.0 * np.pi), 0)
    Qg = np.zeros(1, dtype=np.float64)
    #
    args_Qg = dict()
    args_Qg['nMeasureT'] = nMeasureT
    args_Qg['nMeasureT_times_log2pi'] = nMeasureT_times_log2pi
    #
    args_Qg['Qg'] = Qg
    return args_Qg

def print_args(args, s):
    buffer = 30
    for key in sorted(list(args.keys())):
        key_use = s + key + ':' + ' ' * (buffer - len(key))
        key_type = type(args[key])
        if (key_type == type(dict())):
            print('{:s}:{:s}'.format(key_use, str(key_type)))
            s_use = s + '\t'
            print_args(args[key], s_use)
        elif(key_type == type(torch.Tensor([]))):
            print('{:s}:{:s} (is_cuda = {:s})'.format(key_use, str(key_type), str(args[key].is_cuda)))
        else:
            print('{:s}:{:s}'.format(key_use, str(key_type)))
    return

def convert_dtype(args_in):
    args_out = dict()
    for key in np.sort(list(args_in.keys())):
        key_type = type(args_in[key])
        if (key_type == type(dict())):
            args_out[key] = convert_dtype(args_in[key])
        elif (key_type == type(torch.Tensor([]))):
            key_type_torch = args_in[key].detach().cpu().type()
            # bool
            if (key_type_torch == torch.BoolTensor([]).type()):
                args_out[key] = args_in[key].type(torch.bool)
            # float
            elif (key_type_torch == torch.DoubleTensor([]).type()):
                args_out[key] = args_in[key].type(model.float_type)
            elif (key_type_torch == torch.FloatTensor([]).type()):
                args_out[key] = args_in[key].type(model.float_type)
#             # int
#             elif (key_type_torch == torch.LongTensor([]).type()):
#                 args_out[key] = args_in[key].type(torch.int64)
#             elif (key_type_torch == torch.IntTensor([]).type()):
#                 args_out[key] = args_in[key].type(torch.int32)
#             elif (key_type_torch == torch.ShortTensor([]).type()):
#                 args_out[key] = args_in[key].type(torch.int16)
#             elif (key_type_torch == torch.CarTensor([]).type()):
#                 args_out[key] = args_in[key].type(torch.int8)
#             elif (key_type_torch == torch.ByteTensor([]).type()):
#                 args_out[key] = args_in[key].type(torch.uint8)
            # other
            else:
                args_out[key] = args_in[key]
        else:
            args_out[key] = args_in[key]
    return args_out

def make_args_torch(args):
    args_torch = dict()
    for key in np.sort(list(args.keys())):
        key_type = type(args[key])
        if (key_type == type(dict())):
            args_torch[key] = make_args_gpu(args[key])
        elif (key_type == type(np.array([]))):
            if (args[key].dtype == 'bool'):
                args_torch[key] = torch.from_numpy(args[key].astype(bool))
            else:
                args_torch[key] = torch.from_numpy(args[key]).type(model.float_type)
        elif (key_type == type(torch.Tensor([]))):
            args_torch[key] = args[key].detach()
        elif ((key_type == type(float())) or (key_type == type(np.float64())) or (key_type == type(np.float32())) or (key_type == type(np.float16()))):
#             args_torch[key] = float(args[key])
            args_torch[key] = torch.scalar_tensor(args[key], dtype=model.float_type)
        elif ((key_type == type(int())) or key_type == type(np.int64()) or (key_type == type(np.int32())) or (key_type == type(np.int16()))):
#             args_torch[key] = int(args[key])
            args_torch[key] = torch.scalar_tensor(args[key], dtype=torch.int64)
        elif ((key_type == type(bool())) or (key_type == type(np.bool()))):
            args_torch[key] = torch.scalar_tensor(args[key], dtype=torch.bool)
        else:
            args_torch[key] = args[key]
    return args_torch

def make_args_gpu(args):
    args_gpu = dict()
    for key in np.sort(list(args.keys())):
        key_type = type(args[key])
        if (key_type == type(dict())):
            args_gpu[key] = make_args_gpu(args[key])
        elif (key_type == type(np.array([]))):
            if (args[key].dtype == 'bool'):
                args_gpu[key] = torch.from_numpy(args[key].astype(bool)).cuda()
            else:
                args_gpu[key] = torch.from_numpy(args[key]).type(model.float_type).cuda()
        elif (key_type == type(torch.Tensor([]))):
            args_gpu[key] = args[key].detach().cuda()
        elif ((key_type == type(float())) or (key_type == type(np.float64())) or (key_type == type(np.float32())) or (key_type == type(np.float16()))):
#             args_gpu[key] = float(args[key])
            args_gpu[key] = torch.scalar_tensor(arg_in, dtype=model.float_type).cuda()
        elif ((key_type == type(int())) or (key_type == type(np.int64())) or (key_type == type(np.int32())) or (key_type == type(np.int16()))):
#             args_gpu[key] = int(args[key])
            args_gpu[key] = torch.scalar_tensor(arg_in, dtype=torch.int64).cuda()
        elif ((key_type == type(bool())) or (key_type == type(np.bool()))):
            arg_out = torch.scalar_tensor(arg_in, dtype=torch.bool).cuda()
        else:
            args_gpu[key] = args[key]
    return args_gpu

def convert_to_torch(arg_in):
    key_type = type(arg_in)
    if (key_type == type(np.array([]))):
        if (arg_in.dtype == 'bool'):
            arg_out = torch.from_numpy(arg_in.astype(bool))
        else:
            arg_out = torch.from_numpy(arg_in).type(model.float_type)
    elif (key_type == type(torch.Tensor([]))):
        arg_out = arg_in.detach()
    elif ((key_type == type(float())) or (key_type == type(np.float64())) or (key_type == type(np.float32())) or (key_type == type(np.float16()))):
#         arg_out = float(arg_in)
        arg_out = torch.scalar_tensor(arg_in, dtype=model.float_type)
    elif ((key_type == type(int())) or (key_type == type(np.int32())) or (key_type == type(np.int16()))):
#         arg_out = int(arg_in)
        arg_out = torch.scalar_tensor(arg_in, dtype=torch.int64)
    elif ((key_type == type(bool())) or (key_type == type(np.bool()))):
        arg_out = torch.scalar_tensor(arg_in, dtype=torch.bool)
    else:
        arg_out = arg_in
    return arg_out

def convert_to_gpu(arg_in):
    key_type = type(arg_in)
    if (key_type == type(np.array([]))):
        if (arg_in.dtype == 'bool'):
            arg_out = torch.from_numpy(arg_in.astype(bool)).cuda()
        else:
            arg_out = torch.from_numpy(arg_in).type(model.float_type).cuda()
    elif (key_type == type(torch.Tensor([]))):
        arg_out = arg_in.detach().cuda()
    elif ((key_type == type(float())) or (key_type == type(np.float64())) or (key_type == type(np.float32())) or (key_type == type(np.float16()))):
#         arg_out = float(arg_in)
        arg_out = torch.scalar_tensor(arg_in, dtype=model.float_type).cuda()
    elif ((key_type == type(int())) or key_type == type(np.int64()) or (key_type == type(np.int32())) or (key_type == type(np.int16()))):
#         arg_out = int(arg_in)
        arg_out = torch.scalar_tensor(arg_in, dtype=torch.int64).cuda() # FIXME: determine int type?
    elif ((key_type == type(bool())) or (key_type == type(np.bool()))):
        arg_out = torch.scalar_tensor(arg_in, dtype=torch.bool).cuda()
    else:
        arg_out = arg_in
    return arg_out

def main():
    nT = cfg.nT
    dt = cfg.dt
    use_cuda = cfg.use_cuda
    slow_mode = cfg.slow_mode
    sigma_point_scheme = cfg.sigma_point_scheme
    rand_fac = 1.0 # only used when sigma points are sampled randomly

    # get index of initialization frame & forward frame list
    labelsDLC = np.load(cfg.file_labelsDLC, allow_pickle=True).item()
    frame_list = labelsDLC['frame_list']
    i_frame = np.where(frame_list == cfg.index_frame_ini)[0][0]
    frame_list_fit = np.arange(i_frame,
                               int(i_frame + cfg.dt * cfg.nT),
                               cfg.dt,
                               dtype=np.int64)

    # get arguments
    args_model = helper.get_arguments(cfg.file_origin_coord, cfg.file_calibration, cfg.file_model, cfg.file_labelsDLC,
                                      cfg.scale_factor, cfg.pcutoff)
    args_model['use_custom_clip'] = True

    # numbers
    nMarkers = args_model['numbers']['nMarkers']
    nCameras = args_model['numbers']['nCameras']
    # free parameters
    free_para_bones = args_model['free_para_bones'].cpu().numpy()
    free_para_markers = args_model['free_para_markers'].cpu().numpy()
    free_para_pose = args_model['free_para_pose'].cpu().numpy()
    # bounds
    bounds_free_pose = args_model['bounds_free_pose'].cpu().numpy()
    bounds_free_pose_range = args_model['bounds_free_pose_range'].cpu().numpy()
    bounds_free_pose_0 = args_model['bounds_free_pose_0'].cpu().numpy()

    # remove all free parameters that do not modify the pose
    free_para_bones = np.zeros_like(free_para_bones, dtype=bool)
    free_para_markers = np.zeros_like(free_para_markers, dtype=bool)
    nFree_bones = int(0)
    nFree_markers = int(0)
    args_model['free_para_bones'] = torch.from_numpy(free_para_bones)
    args_model['free_para_markers'] = torch.from_numpy(free_para_markers)
    args_model['nFree_bones'] = nFree_bones
    args_model['nFree_markers'] = nFree_markers

    # create correct free_para
    free_para = np.concatenate([free_para_bones,
                                free_para_markers,
                                free_para_pose], 0)


    # dim_z / dim_x
    dim_z = int(np.sum(free_para))
    dim_z2 = 2 * dim_z
    dim_x = nCameras * nMarkers * 2

    # measure / measure_mask
    measure = args_model['labels'][frame_list_fit].clone()
    measure_mask = args_model['labels_mask'][frame_list_fit].clone()
    #
    # normalize the measurements
    measure[:, :, :, 0] = measure[:, :, :, 0] / (cfg.normalize_camera_sensor_x * 0.5) - 1.0
    measure[:, :, :, 1] = measure[:, :, :, 1] / (cfg.normalize_camera_sensor_y * 0.5) - 1.0
    #
    measure = measure[:, :, :, :2].reshape(nT, dim_x)
    measure_mask = measure_mask.reshape(nT, nCameras, nMarkers, 1)
    measure_mask = measure_mask.repeat(1, 1, 1, 2)
    measure_mask = measure_mask.reshape(nT, dim_x)
    #
    measure_mask_exclude = torch.ones(dim_x, dtype=torch.bool)
    measure_mask_exclude = torch.any(measure_mask, 0)
#     measure_mask_exclude = (torch.sum(measure_mask, 0) >= max(1, nT*0.00))
    #
    measure = measure[:, measure_mask_exclude]
    measure_mask = measure_mask[:, measure_mask_exclude]
    #
    dim_x = int(torch.sum(measure_mask_exclude))
    #
#     measure_mask_sum = torch.zeros(1, dtype=torch.int64)
#     measure_mask_outer = torch.zeros((dim_x, dim_x), dtype=torch.bool)
    #
    nMeasure = torch.sum(measure_mask)

    # get number of free parameters
    nFree_mu0 = int(dim_z)
    nFree_var0 = int((dim_z**2+dim_z)/2)
#     nFree_A = int(dim_z**2)
    nFree_A = 0 # when A is assumed to be constant
    nFree_var_f = int((dim_z**2+dim_z)/2)
    nFree_var_g = int(dim_x) # only entries on the diagonal
    nFree = nFree_mu0 + nFree_var0 + nFree_A + nFree_var_f + nFree_var_g



    # mu0_ini
    # load from ini
    mu_ini = np.load(cfg.folder_init + '/x_ini.npy', allow_pickle=True)
    #
    # ATTENTION: [mu_ini_min <= mu_ini <= mu_ini_max] is enforced in initialization.py
    mu_ini_fac = 0.9
    mu_ini_min = (args_model['bounds_free_pose_0'] - args_model['bounds_free_pose_range'] * mu_ini_fac).numpy()
    mu_ini_max = (args_model['bounds_free_pose_0'] + args_model['bounds_free_pose_range'] * mu_ini_fac).numpy()
    mu_ini_free_clamped = np.copy(mu_ini[free_para])
    mask_down = (mu_ini_free_clamped < mu_ini_min)
    mask_up = (mu_ini_free_clamped > mu_ini_max)
    mu_ini_free_clamped[mask_down] = mu_ini_min[mask_down]
    mu_ini_free_clamped[mask_up] = mu_ini_max[mask_up]
    mu_ini[free_para] = np.copy(mu_ini_free_clamped)
    # initialize args regarding x
    args_model['x_torch'] = torch.from_numpy(mu_ini).type(model.float_type)
    args_model['x_free_torch'] = torch.from_numpy(mu_ini[free_para]).type(model.float_type)

    # set initial values for the model parameters
    mu0 = model.do_normalization(torch.from_numpy(mu_ini[free_para][None, :]), args_model)[0].numpy()
    var0 = np.identity(dim_z, dtype=np.float64) * cfg.noise
    A = np.identity(dim_z, dtype=np.float64)
    var_f = np.identity(dim_z, dtype=np.float64) * cfg.noise
    var_g = np.identity(dim_x, dtype=np.float64) * cfg.noise

    #
    mu0 = torch.from_numpy(mu0).type(model.float_type)
    var0 = torch.from_numpy(var0).type(model.float_type)
    A = torch.from_numpy(A).type(model.float_type)
    var_f = torch.from_numpy(var_f).type(model.float_type)
    var_g = torch.from_numpy(var_g).type(model.float_type)

    #
    dim_z2 = 2 * dim_z
    if (sigma_point_scheme == 3):
        nSigmaPoints = 2*dim_z + 1
        nSigmaPoints2 = 2*dim_z2 + 1
    elif (sigma_point_scheme == 5):
        nSigmaPoints = 2*dim_z**2 + 1
        nSigmaPoints2 = 2*dim_z2**2 + 1
    elif (sigma_point_scheme == 0):
        nSigmaPoints = int(rand_fac * 2*dim_z**2) + 1 # should be uneven
        nSigmaPoints2 = int(rand_fac * 2*dim_z2**2) + 1 # should be uneven


    # set
    w_m = np.full(nSigmaPoints, 0.0, dtype=np.float64)
    w_c = np.full(nSigmaPoints, 0.0, dtype=np.float64)
    w_m2 = np.full(nSigmaPoints2, 0.0, dtype=np.float64)
    w_c2 = np.full(nSigmaPoints2, 0.0, dtype=np.float64)
    alpha = 1.0
    beta = 0.0
    kappa = 0.0
    lamb = alpha**2 * (dim_z + kappa) - dim_z
    lamb2 = alpha**2 * (dim_z2 + kappa) - dim_z2
    w_m[0] = lamb / (dim_z + lamb)
    w_m2[0] = lamb2 / (dim_z2 + lamb2)
    w_m[1:] = 1.0 / (2.0 * (dim_z + lamb))
    w_m2[1:] = 1.0 / (2.0 * (dim_z2 + lamb2))
    w_c[0] = lamb / (dim_z + lamb) + (1.0 - alpha**2 + beta)
    w_c2[0] = lamb2 / (dim_z2 + lamb2) + (1.0 - alpha**2 + beta)
    w_c[1:] = 1.0 / (2.0 * (dim_z + lamb))
    w_c2[1:] = 1.0 / (2.0 * (dim_z2 + lamb2))
    sqrt_dimZ_p_lamb = np.sqrt(dim_z + lamb)
    sqrt_dimZ_p_lamb2 = np.sqrt(dim_z2 + lamb2)


    # for ukf5 (w1 = 2**0.5 * w2)
    if (sigma_point_scheme == 5):
        print('ATTENTION: USING UKF5')
        w_m = np.full(nSigmaPoints, 0.0, dtype=np.float64)
        w_c = np.full(nSigmaPoints, 0.0, dtype=np.float64)
        w_m2 = np.full(nSigmaPoints2, 0.0, dtype=np.float64)
        w_c2 = np.full(nSigmaPoints2, 0.0, dtype=np.float64)
        n1 = 2*dim_z
        n2 = 2*dim_z**2 - n1
        n1_2 = 2*dim_z2
        n2_2 = 2*dim_z2**2 - n1_2
        w_m[0] = 1.0 / (n1 + 2.0**-0.5 * n2 + 1.0)
        w_m2[0] = 1.0 / (n1_2 + 2.0**-0.5 * n2_2 + 1.0)
        w_m[1:2*dim_z+1] = (1.0 - w_m[0]) / (n1 + 2.0**-0.5 * n2)
        w_m2[1:2*dim_z2+1] = (1.0 - w_m2[0]) / (n1_2 + 2.0**-0.5 * n2_2)
        w_m[2*dim_z+1:] = ((1.0 - w_m[0]) / (n1 + 2.0**-0.5 * n2)) * 2**-0.5
        w_m2[2*dim_z2+1:] = ((1.0 - w_m2[0]) / (n1_2 + 2.0**-0.5 * n2_2)) * 2**-0.5
        w_c = np.copy(w_m)
        w_c2 = np.copy(w_m2)
        sqrt_dimZ_p_lamb = np.sqrt(dim_z / (1.0 - 1.0/(2*dim_z + 1))) / 2**0.5
        sqrt_dimZ_p_lamb2 = np.sqrt(dim_z2 / (1.0 - 1.0/(2*dim_z2 + 1))) / 2**0.5

    #
    w_m = torch.from_numpy(w_m).type(model.float_type)
    w_c = torch.from_numpy(w_c).type(model.float_type)
    w_m2 = torch.from_numpy(w_m2).type(model.float_type)
    w_c2 = torch.from_numpy(w_c2).type(model.float_type)
    #
    print('w_m:')
    print(w_m)
    print(float(torch.sum(w_m)))
    print('w_m2:')
    print(w_m2)
    print(float(torch.sum(w_m2)))
    print('w_c:')
    print(w_c)
    print(float(torch.sum(w_c)))
    print('w_c2:')
    print(w_c2)
    print(float(torch.sum(w_c2)))
    print('sqrt_dimZ_p_lamb:')
    print(sqrt_dimZ_p_lamb)
    print('sqrt_dimZ_p_lamb2:')
    print(sqrt_dimZ_p_lamb2)
    print()
    print('dim_z:\t\t{:06d}'.format(dim_z))
    print('dim_x:\t\t{:06d}'.format(dim_x))
    print('nFree (mu0):\t{:06d}'.format(nFree_mu0))
    print('nFree (var0):\t{:06d}'.format(nFree_var0))
    print('nFree (A):\t{:06d}'.format(nFree_A))
    print('nFree (var_f):\t{:06d}'.format(nFree_var_f))
    print('nFree (var_g):\t{:06d}'.format(nFree_var_g))
    print('nFree (total):\t{:06d}'.format(nFree))
    print('nMeasure:\t{:06d}'.format(nMeasure))
    print()

    if ((abs(float(torch.sum(w_m)) - 1.0) >= 2**-23) or (abs(float(torch.sum(w_m2)) - 1.0) >= 2**-23)):
        print('ERROR: w_m or w_m2 does not add up to 1.0')
        raise
    if ((abs(float(torch.sum(w_c)) - 1.0) >= 2**-23) or (abs(float(torch.sum(w_c2)) - 1.0) >= 2**-23)):
        print('ERROR: w_c or w_c2 does not add up to 1.0')
        raise

    # kalman / em
    args_kalman = dict()
    #
    args_kalman['slow_mode'] = bool(slow_mode)
    #
    args_kalman['nT'] = convert_to_torch(nT)
    args_kalman['dim_z'] = convert_to_torch(dim_z)
    args_kalman['dim_z2'] = convert_to_torch(dim_z2) # for em
    args_kalman['dim_x'] = convert_to_torch(dim_x) # for em
    args_kalman['sigma_point_scheme'] = convert_to_torch(sigma_point_scheme)
    args_kalman['nSigmaPoints'] = convert_to_torch(nSigmaPoints) # for em
    args_kalman['w_m'] = w_m
    args_kalman['w_c'] = w_c
    args_kalman['sqrt_dimZ_p_lamb'] = convert_to_torch(sqrt_dimZ_p_lamb)
    args_kalman['nSigmaPoints2'] = convert_to_torch(nSigmaPoints2) # for em
    args_kalman['w_m2'] = w_m2 # for em
    args_kalman['w_c2'] = w_c2 # for em
    args_kalman['sqrt_dimZ_p_lamb2'] = convert_to_torch(sqrt_dimZ_p_lamb2) # for em
    args_kalman['measure'] = measure
    args_kalman['measure_mask'] = measure_mask
    args_kalman['measure_mask_exclude'] = measure_mask_exclude
    #
    args_kalman['sigmaPoints_kalman'] = torch.zeros((nSigmaPoints, dim_z), dtype=model.float_type)
    args_kalman['mu_kalman'] = torch.zeros((nT+1, dim_z), dtype=model.float_type)
    args_kalman['var_kalman'] = torch.zeros((nT+1, dim_z, dim_z), dtype=model.float_type)
    args_kalman['G_kalman'] = torch.zeros((nT, dim_z, dim_z), dtype=model.float_type)
    # ukf
    args_kalman['var_p2_ukf'] = torch.zeros((nSigmaPoints, dim_z), dtype=model.float_type)
    args_kalman['var_p31_ukf'] = torch.zeros(dim_z, dtype=model.float_type)
    args_kalman['var_p32_ukf'] = torch.zeros((dim_z, dim_z), dtype=model.float_type)
    args_kalman['var_u2_ukf'] = torch.zeros((nSigmaPoints, dim_x), dtype=model.float_type)
    args_kalman['var_u31_ukf'] = torch.zeros(dim_x, dtype=model.float_type)
    args_kalman['var_u32_ukf'] = torch.zeros((dim_x, dim_x), dtype=model.float_type)
    args_kalman['var_u33_ukf'] = torch.zeros((dim_z, dim_x), dtype=model.float_type)
    args_kalman['var_u41_ukf'] = torch.zeros((dim_z, dim_x), dtype=model.float_type)
    # uks
    args_kalman['var_2_uks'] = torch.zeros((nSigmaPoints, dim_z), dtype=model.float_type)
    args_kalman['var_31_uks'] = torch.zeros(dim_z, dtype=model.float_type)
    args_kalman['var_32_uks'] = torch.zeros((dim_z, dim_z), dtype=model.float_type)
    args_kalman['var_33_uks'] = torch.zeros((dim_z, dim_z), dtype=model.float_type)
    #
    if slow_mode:
        args_kalman['outer_z'] = torch.zeros((dim_z, dim_z), dtype=model.float_type)
        args_kalman['var_pairwise'] = torch.zeros((dim_z2, dim_z2), dtype=model.float_type)
        args_kalman['mu_pairwise'] = torch.zeros(dim_z2, dtype=model.float_type)
        args_kalman['sigma_points'] = torch.zeros((nSigmaPoints, dim_z), dtype=model.float_type) # for em
        args_kalman['sigma_points_g'] = torch.zeros((nSigmaPoints, dim_x), dtype=model.float_type) # for em
        args_kalman['sigma_points_pairwise'] = torch.zeros((nSigmaPoints2, dim_z2), dtype=model.float_type) # for em
        args_kalman['x1_m_fx0'] = torch.zeros((nSigmaPoints2, dim_z), dtype=model.float_type) # for em
    else:
        args_kalman['outer_z'] = torch.zeros((nT, dim_z, dim_z), dtype=model.float_type)
        args_kalman['var_pairwise'] = torch.zeros((nT, dim_z2, dim_z2), dtype=model.float_type)
        args_kalman['mu_pairwise'] = torch.zeros((nT, dim_z2), dtype=model.float_type)
        args_kalman['sigma_points'] = torch.zeros((nT, nSigmaPoints, dim_z), dtype=model.float_type) # for em
        args_kalman['sigma_points_g'] = torch.zeros((nT, nSigmaPoints, dim_x), dtype=model.float_type) # for em
        args_kalman['sigma_points_pairwise'] = torch.zeros((nT, nSigmaPoints2, dim_z2), dtype=model.float_type) # for em
        args_kalman['x1_m_fx0'] = torch.zeros((nT, nSigmaPoints2, dim_z), dtype=model.float_type) # for em
    args_kalman['measurement_expectation0'] = torch.zeros((nT, dim_x), dtype=model.float_type) # for em
    args_kalman['measurement_expectation1'] = torch.zeros((nT, dim_x), dtype=model.float_type) # for em
    args_kalman['y1_m_hx1_2'] = torch.zeros((nT, dim_x), dtype=model.float_type)# for em


    # all
    args = dict()
    args['use_cuda'] = False

    # plot
    args_model['plot'] = False
    #
    args['args_model'] = args_model
    args['args_kalman'] = args_kalman

    # args_Q
    args_Qg = gen_args_Qg(args)
    args_Qg = make_args_torch(args_Qg)
    args_kalman['args_Qg'] = args_Qg # for em
    args['args_kalman'] = args_kalman

    # args_model
    if (use_cuda):
        args_gpu = dict()
        args_gpu['use_cuda'] = convert_to_gpu(use_cuda)
        #
        args_model_gpu = dict()
        # model
        model_dict = args_model['model']
        model_dict_gpu = dict()
        model_dict_gpu['skeleton_vertices'] = convert_to_gpu(model_dict['skeleton_vertices'])
        model_dict_gpu['skeleton_vertices_new'] = convert_to_gpu(model_dict['skeleton_vertices_new'])
        model_dict_gpu['skeleton_edges'] = convert_to_gpu(model_dict['skeleton_edges'])
        model_dict_gpu['bone_lengths'] = convert_to_gpu(model_dict['bone_lengths'])
        model_dict_gpu['bone_lengths_index'] = convert_to_gpu(model_dict['bone_lengths_index'])
        model_dict_gpu['skeleton_vertices_links'] = convert_to_gpu(model_dict['skeleton_vertices_links'])
        model_dict_gpu['joint_marker_vectors'] = convert_to_gpu(model_dict['joint_marker_vectors'])
        model_dict_gpu['joint_marker_vectors_new'] = convert_to_gpu(model_dict['joint_marker_vectors_new'])
        model_dict_gpu['skeleton_coords_index'] = convert_to_gpu(model_dict['skeleton_coords_index'])
        model_dict_gpu['joint_marker_index'] = convert_to_gpu(model_dict['joint_marker_index'])
        model_dict_gpu['skeleton_coords'] = convert_to_gpu(model_dict['skeleton_coords'])
        model_dict_gpu['skeleton_coords0'] = convert_to_gpu(model_dict['skeleton_coords0'])
        model_dict_gpu['I_bone'] = convert_to_gpu(model_dict['I_bone'])
        #
        model_dict_gpu['is_euler'] = convert_to_gpu(model_dict['is_euler'])
        #
        args_model_gpu['model'] = model_dict_gpu
        # calibration
        calibration_dict = args_model['calibration']
        calibration_dict_gpu = dict()
        calibration_dict_gpu['A_fit'] = convert_to_gpu(calibration_dict['A_fit'])
        calibration_dict_gpu['k_fit'] = convert_to_gpu(calibration_dict['k_fit'])
        calibration_dict_gpu['RX1_fit'] = convert_to_gpu(calibration_dict['RX1_fit'])
        calibration_dict_gpu['tX1_fit'] = convert_to_gpu(calibration_dict['tX1_fit'])
        args_model_gpu['calibration'] = calibration_dict_gpu
        # numbers
        numbers_dict = args_model['numbers']
        numbers_dict_gpu = dict()
        numbers_dict_gpu['nBones'] = convert_to_gpu(numbers_dict['nBones'])
        numbers_dict_gpu['nMarkers'] = convert_to_gpu(numbers_dict['nMarkers'])
        numbers_dict_gpu['nCameras'] = convert_to_gpu(numbers_dict['nCameras'])
        args_model_gpu['numbers'] = numbers_dict_gpu
        #
        args_model_gpu['free_para_bones'] = convert_to_gpu(args_model['free_para_bones'])
        args_model_gpu['bounds_bones'] = convert_to_gpu(args_model['bounds_bones'])
        args_model_gpu['bounds_free_bones'] = convert_to_gpu(args_model['bounds_free_bones'])
        args_model_gpu['nPara_bones'] = convert_to_gpu(args_model['nPara_bones'])
        args_model_gpu['nFree_bones'] = convert_to_gpu(args_model['nFree_bones'])
        args_model_gpu['free_para_markers'] = convert_to_gpu(args_model['free_para_markers'])
        args_model_gpu['bounds_markers'] = convert_to_gpu(args_model['bounds_markers'])
        args_model_gpu['bounds_free_markers'] = convert_to_gpu(args_model['bounds_free_markers'])
        args_model_gpu['nPara_markers'] = convert_to_gpu(args_model['nPara_markers'])
        args_model_gpu['nFree_markers'] = convert_to_gpu(args_model['nFree_markers'])
        args_model_gpu['free_para_pose'] = convert_to_gpu(args_model['free_para_pose'])
        args_model_gpu['bounds_pose'] = convert_to_gpu(args_model['bounds_pose'])
        args_model_gpu['nPara_pose'] = convert_to_gpu(args_model['nPara_pose'])
        args_model_gpu['nFree_pose'] = convert_to_gpu(args_model['nFree_pose'])
        args_model_gpu['bounds_free_pose'] = convert_to_gpu(args_model['bounds_free_pose'])
        args_model_gpu['bounds_free_pose_range'] = convert_to_gpu(args_model['bounds_free_pose_range'])
        args_model_gpu['bounds_free_pose_0'] = convert_to_gpu(args_model['bounds_free_pose_0'])
        args_model_gpu['is_euler'] = convert_to_gpu(args_model['is_euler'])
        #
        args_model_gpu['x_torch'] = convert_to_gpu(args_model['x_torch'])
        #
        args_model_gpu['plot'] = convert_to_gpu(args_model['plot'])
        #
        args_gpu['args_model'] = args_model_gpu
        #
        #
        # args_kalman
        args_kalman_gpu = dict()
        #
        args_kalman_gpu['slow_mode'] = convert_to_gpu(args_kalman['slow_mode'])
        #
        args_kalman_gpu['nT'] = convert_to_gpu(args_kalman['nT'])
        args_kalman_gpu['dim_z'] = convert_to_gpu(args_kalman['dim_z'])
        args_kalman_gpu['sigma_point_scheme'] = convert_to_gpu(args_kalman['sigma_point_scheme'])
        args_kalman_gpu['w_m'] = convert_to_gpu(args_kalman['w_m'])
        args_kalman_gpu['w_c'] = convert_to_gpu(args_kalman['w_c'])
        args_kalman_gpu['sqrt_dimZ_p_lamb'] = convert_to_gpu(args_kalman['sqrt_dimZ_p_lamb'])
        args_kalman_gpu['measure'] = convert_to_gpu(args_kalman['measure'])
        args_kalman_gpu['measure_mask'] = convert_to_gpu(args_kalman['measure_mask'])
        args_kalman_gpu['measure_mask_exclude'] = convert_to_gpu(args_kalman['measure_mask_exclude'])
        args_kalman_gpu['dim_z2'] = convert_to_gpu(args_kalman['dim_z2']) # for em
        args_kalman_gpu['dim_x'] = convert_to_gpu(args_kalman['dim_x']) # for em
        args_kalman_gpu['nSigmaPoints'] = convert_to_gpu(args_kalman['nSigmaPoints']) # for em
        args_kalman_gpu['nSigmaPoints2'] = convert_to_gpu(args_kalman['nSigmaPoints2']) # for em
        args_kalman_gpu['w_m2'] = convert_to_gpu(args_kalman['w_m2']) # for em
        args_kalman_gpu['w_c2'] = convert_to_gpu(args_kalman['w_c2']) # for em
        args_kalman_gpu['sqrt_dimZ_p_lamb2'] = convert_to_gpu(args_kalman['sqrt_dimZ_p_lamb2']) # for em
        args_kalman_gpu['args_Qg'] = make_args_gpu(args_kalman['args_Qg']) # for em
        #
        args_kalman_gpu['sigmaPoints_kalman'] = convert_to_gpu(args_kalman['sigmaPoints_kalman'])
        args_kalman_gpu['mu_kalman'] = convert_to_gpu(args_kalman['mu_kalman'])
        args_kalman_gpu['var_kalman'] = convert_to_gpu(args_kalman['var_kalman'])
        args_kalman_gpu['G_kalman'] = convert_to_gpu(args_kalman['G_kalman'])
        args_kalman_gpu['var_p2_ukf'] = convert_to_gpu(args_kalman['var_p2_ukf'])
        args_kalman_gpu['var_p31_ukf'] = convert_to_gpu(args_kalman['var_p31_ukf'])
        args_kalman_gpu['var_p32_ukf'] = convert_to_gpu(args_kalman['var_p32_ukf'])
        args_kalman_gpu['var_u2_ukf'] = convert_to_gpu(args_kalman['var_u2_ukf'])
        args_kalman_gpu['var_u31_ukf'] = convert_to_gpu(args_kalman['var_u31_ukf'])
        args_kalman_gpu['var_u32_ukf'] = convert_to_gpu(args_kalman['var_u32_ukf'])
        args_kalman_gpu['var_u33_ukf'] = convert_to_gpu(args_kalman['var_u33_ukf'])
        args_kalman_gpu['var_u41_ukf'] = convert_to_gpu(args_kalman['var_u41_ukf'])
        args_kalman_gpu['var_2_uks'] = convert_to_gpu(args_kalman['var_2_uks'])
        args_kalman_gpu['var_31_uks'] = convert_to_gpu(args_kalman['var_31_uks'])
        args_kalman_gpu['var_32_uks'] = convert_to_gpu(args_kalman['var_32_uks'])
        args_kalman_gpu['var_33_uks'] = convert_to_gpu(args_kalman['var_33_uks'])
        #
        args_kalman_gpu['measurement_expectation0'] = convert_to_gpu(args_kalman['measurement_expectation0']) # for em
        args_kalman_gpu['measurement_expectation1'] = convert_to_gpu(args_kalman['measurement_expectation1']) # for em
        args_kalman_gpu['outer_z'] = convert_to_gpu(args_kalman['outer_z'])
        args_kalman_gpu['var_pairwise'] = convert_to_gpu(args_kalman['var_pairwise'])
        args_kalman_gpu['mu_pairwise'] = convert_to_gpu(args_kalman['mu_pairwise'])
        args_kalman_gpu['sigma_points'] = convert_to_gpu(args_kalman['sigma_points']) # for em
        args_kalman_gpu['sigma_points_g'] = convert_to_gpu(args_kalman['sigma_points_g']) # for em
        args_kalman_gpu['sigma_points_pairwise'] = convert_to_gpu(args_kalman['sigma_points_pairwise']) # for em
        args_kalman_gpu['x1_m_fx0'] = convert_to_gpu(args_kalman['x1_m_fx0']) # for em
        args_kalman_gpu['y1_m_hx1_2'] = convert_to_gpu(args_kalman['y1_m_hx1_2']) # for em
        #
        args_gpu['args_kalman'] = args_kalman_gpu
        #
        args_gpu = convert_dtype(args_gpu)
        print_args(args_gpu, '')
        #
        #
        mu0 = mu0.cuda()
        var0 = var0.cuda()
        A = A.cuda()
        var_f = var_f.cuda()
        var_g = var_g.cuda()

    # to run EM algorithm
    mu0_ini = mu0.clone()
    var0_ini = var0.clone()
    A_ini = A.clone()
    var_f_ini = var_f.clone()
    var_g_ini = var_g.clone()
    if (use_cuda):
        mu0, var0, A, var_f, var_g = em.run(mu0_ini, var0_ini, A_ini, var_f_ini, var_g_ini,
                                            args_gpu)
        mu_uks, var_uks, _ = kalman.uks(mu0, var0, A, var_f, var_g, args_gpu)
    else:
        mu0, var0, A, var_f, var_g = em.run(mu0_ini, var0_ini, A_ini, var_f_ini, var_g_ini,
                                            args)
        mu_uks, var_uks, _ = kalman.uks(mu0, var0, A, var_f, var_g, args)
    save_dict = np.load(os.path.join(cfg.folder_save,'save_dict.npy'), allow_pickle=True).item()
    save_dict['mu0'] = mu0.detach().cpu().numpy()
    save_dict['var0'] = var0.detach().cpu().numpy()
    save_dict['A'] = A.detach().cpu().numpy()
    save_dict['var_f'] = var_f.detach().cpu().numpy()
    save_dict['var_g'] = var_g.detach().cpu().numpy()
    save_dict['mu_uks'] = mu_uks.detach().cpu().numpy()
    save_dict['var_uks'] = var_uks.detach().cpu().numpy()
    np.save(os.path.join(cfg.folder_save,'save_dict.npy'), save_dict)

    # to get 3D joint & marker locations
    args['args_model']['plot'] = True
    marker_proj, marker_pos, skel_pos = model.fcn_emission_free(mu_uks[1:], args['args_model'])
    marker_proj = marker_proj.detach().cpu().numpy().reshape(nT, nCameras, nMarkers, 2)
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

if __name__ == '__main__':
    main()
