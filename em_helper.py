#!/usr/bin/env python3

import numpy as np
import torch
import time

import configuration as cfg
import em
import helper
import kalman
import model


def gen_args_Qg(args):
    args_kalman = args['args_kalman']
    nT = args_kalman['nT']
    dim_x = args_kalman['dim_x']
    nSigmaPoints = args_kalman['nSigmaPoints']
    w_m = args_kalman['w_m']
    measure = args_kalman['measure'].cpu().numpy()
    measure_mask = args_kalman['measure_mask'].cpu().numpy().astype(bool)
    #
    not_measure_mask = ~measure_mask
    #
    measure_use = np.tile(measure, nSigmaPoints)
    measure_use = np.reshape(measure_use, (nT, nSigmaPoints, dim_x))
    #
    nMeasure = np.sum(measure_mask).astype(np.float64)
    nMeasure_times_log2pi = nMeasure * np.log(2.0 * np.pi)
    nMeasure_t = np.sum(measure_mask, 0).astype(np.float64)
    #
    trace_diag_elements = np.zeros((nT, dim_x), dtype=np.float64)
    Qg = np.zeros(1, dtype=np.float64)
    grad = np.zeros(dim_x, dtype=np.float64)
    #
    #
    args_Qg = dict()
    args_Qg['w_m'] = w_m
    args_Qg['not_measure_mask'] = not_measure_mask
    args_Qg['measure_use'] = measure_use
    args_Qg['nMeasure_times_log2pi'] = nMeasure_times_log2pi
    args_Qg['nMeasure_t'] = nMeasure_t
    #
    args_Qg['trace_diag_elements'] = trace_diag_elements
    args_Qg['Qg'] = Qg
    args_Qg['grad'] = grad
    return args_Qg

def print_args(args, s):
    buffer = 30
    for key in np.sort(list(args.keys())):
        key_use = s + key + ':' + ' ' * (buffer - len(key))
        key_type = type(args[key])
        if (key_type == type(dict())):
            print('{:s}:{:s}'.format(key_use, str(key_type)))
            s_use = s + '\t'
            print_args(args[key], s_use)
        elif(key_type == type(torch.Tensor([]))):
            print('{:s}:{:s} (is_cuda = {:s})'.format(key_use, str(key_type), str(args[key].is_cuda)))
        else:
            print('{:s}:'.format(key_use, str(key_type)))
    return

def convert_dtype(args_in):
    args_out = dict()
    for key in np.sort(list(args_in.keys())):
        key_type = type(args_in[key])
        if (key_type == type(dict())):
            args_out[key] = convert_dtype(args_in[key])
        elif (key_type == type(torch.Tensor([]))):
            args_out[key] = args_in[key].half()
#         elif (key_type == type(float())):
#             args_out[key] = torch.float16(args_in[key])
#         elif (key_type == type(int())):
#             args_out[key] = torch.int16(args_in[key])
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
            arg_use = args[key]
            if (args[key].dtype == 'bool'):
                arg_use = arg_use.astype(np.uint8)
            args_torch[key] = torch.from_numpy(arg_use)
        elif (key_type == type(torch.Tensor([]))):
            args_torch[key] = args[key].detach()
        elif (key_type == type(np.float64())):
            args_torch[key] = float(args[key])
        elif (key_type == type(np.int64())): 
            args_torch[key] = int(args[key])
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
            arg_use = args[key]
            if (args[key].dtype == 'bool'):
                arg_use = arg_use.astype(np.uint8)
            args_gpu[key] = torch.from_numpy(arg_use).cuda()
        elif (key_type == type(torch.Tensor([]))):
            args_gpu[key] = args[key].detach().cuda()
        elif (key_type == type(np.float64())):
            args_gpu[key] = float(args[key])
        elif (key_type == type(np.int64())): 
            args_gpu[key] = int(args[key])
        else:
            args_gpu[key] = args[key]
    return args_gpu
    
def convert_to_torch(arg_in):
    key_type = type(arg_in)
    if (key_type == type(np.array([]))):
        arg_out = arg_in
        if (arg_out.dtype == 'bool'):
            arg_out = arg_use.astype(np.uint8)
        arg_out = torch.from_numpy(arg_use)
    elif (key_type == type(torch.Tensor([]))):
        arg_out = arg_in.detach()
    elif (key_type == type(np.float64())):
        arg_out = float(arg_in)
    elif (key_type == type(np.int64())): 
        arg_out = int(arg_in)
    else:
        arg_out = arg_in
    return arg_out
    
def convert_to_gpu(arg_in):
    key_type = type(arg_in)
    if (key_type == type(np.array([]))):
        arg_out = arg_in
        if (arg_out.dtype == 'bool'):
            arg_out = arg_use.astype(np.uint8)
        arg_out = torch.from_numpy(arg_use).cuda()
    elif (key_type == type(torch.Tensor([]))):
        arg_out = arg_in.detach().cuda()
    elif (key_type == type(np.float64())):
        arg_out = float(arg_in)
    elif (key_type == type(np.int64())): 
        arg_out = int(arg_in)
    else:
        arg_out = arg_in
    return arg_out