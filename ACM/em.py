#!/usr/bin/env python3

import math
import numpy as np
import torch
import os

import configuration as cfg

from . import kalman
from . import model

# IMPROVEMENT: put arrays of args_Qg in args
# IMPROVEMENT: put arrays of the third row in args
# IMPROVEMENT: also implement learning of A in slow mode
def update_theta(mu_uks, var_uks, G_uks,
                 args, args_Qg,
                 x1_m_fx0, y1_m_hx1_2, measurement_expectation,
                 var_f_1, var_g_1_L, A_1):
    args_kalman = args['args_kalman']
    
    slow_mode = args_kalman['slow_mode']
    nT = args_kalman['nT']
    dim_z = args_kalman['dim_z']
    dim_x = args_kalman['dim_x'] # for fast mode
    sigma_point_scheme = args_kalman['sigma_point_scheme']
    w_c = args_kalman['w_c'] # for covariance terms
    w_c2 = args_kalman['w_c2'] # for covariance terms
    nSigmaPoints = args_kalman['nSigmaPoints']
    nSigmaPoints2 = args_kalman['nSigmaPoints2']
    sqrt_dimZ_p_lamb = args_kalman['sqrt_dimZ_p_lamb']
    sqrt_dimZ_p_lamb2 = args_kalman['sqrt_dimZ_p_lamb2']

    measure = args_kalman['measure']
    measure_mask = args_kalman['measure_mask']
    measure_mask_exclude = args_kalman['measure_mask_exclude']
    args_model = args['args_model']
    
    outer_z = args_kalman['outer_z']
    var_pairwise = args_kalman['var_pairwise']   
    mu_pairwise = args_kalman['mu_pairwise']
    sigma_points = args_kalman['sigma_points']
    sigma_points_g = args_kalman['sigma_points_g']
    sigma_points_pairwise = args_kalman['sigma_points_pairwise']
    
    nMeasureT = args_Qg['nMeasureT']
    
    # zero arrays to make sure no old values are still there (should not be necessary)
    x1_m_fx0.data.zero_()
    y1_m_hx1_2.data.zero_()
    measurement_expectation.data.zero_()
    var_f_1.data.zero_()
    var_g_1_L.data.zero_()
    if slow_mode:
        for t in range(1, nT+1):
            # sigma points
            if (sigma_point_scheme == 3):
                sigma_points.data.copy_(kalman.calc_sigma_points_p3(mu_uks[t], sqrt_dimZ_p_lamb, kalman.cholesky_save(var_uks[t])).data)
            elif (sigma_point_scheme == 5):
                sigma_points.data.copy_(kalman.calc_sigma_points_p5(mu_uks[t], sqrt_dimZ_p_lamb, kalman.cholesky_save(var_uks[t])).data)
            elif (sigma_point_scheme == 0):
                sigma_points.data.copy_(kalman.calc_sigma_points_rand(mu_uks[t], sqrt_dimZ_p_lamb, kalman.cholesky_save(var_uks[t]),
                                                                      nSigmaPoints).data)
            sigma_points_g.data.copy_((model.fcn_emission_free(sigma_points, args_model)[:, measure_mask_exclude]).data)
            # pairwise sigma points
            outer_z.data.copy_(torch.mm(G_uks[t-1], var_uks[t]).data)
            var_pairwise.data.copy_(torch.cat([torch.cat([var_uks[t], outer_z.transpose(1, 0)], 1),
                                               torch.cat([outer_z, var_uks[t-1]], 1)], 0).data)
            mu_pairwise.data.copy_(torch.cat([mu_uks[t], mu_uks[t-1]], 0).data)
            if (sigma_point_scheme == 3):
                sigma_points_pairwise.data.copy_(kalman.calc_sigma_points_p3(mu_pairwise, sqrt_dimZ_p_lamb2, kalman.cholesky_save(var_pairwise)).data)
            elif (sigma_point_scheme == 5):
                sigma_points_pairwise.data.copy_(kalman.calc_sigma_points_p5(mu_pairwise, sqrt_dimZ_p_lamb2, kalman.cholesky_save(var_pairwise)).data)
            elif (sigma_point_scheme == 0):
                sigma_points_pairwise.data.copy_(kalman.calc_sigma_points_rand(mu_pairwise, sqrt_dimZ_p_lamb2, kalman.cholesky_save(var_pairwise),
                                                                               nSigmaPoints2).data)
            x1_m_fx0.data.copy_(kalman.substraction_save(sigma_points_pairwise[:, :dim_z],
                                                         sigma_points_pairwise[:, dim_z:]).data)
            var_f_1.data.add_(torch.sum((w_c2[:, None, None] / float(nT)) * \
                                        torch.einsum('mi,mj->mij',
                                                     (x1_m_fx0, x1_m_fx0)), 0).data)
            var_f_1.data.copy_((0.5 * (var_f_1 + var_f_1.transpose(1, 0))).data)
            # var_g
            y1_m_hx1_2[t-1].data.copy_(torch.sum(w_c[:, None] * \
                                                 kalman.substraction_save(measure[t-1, None, :],
                                                                          sigma_points_g)**2, 0).data)
            y1_m_hx1_2[t-1, ~measure_mask[t-1]] = 0.0
            #
            measurement_expectation[t-1].data.copy_(sigma_points_g[0, :].data)
        var_g_1_L.data.copy_((torch.sum(y1_m_hx1_2, 0) * nMeasureT**-1).data)        
    else:
        # sigma points
        if (sigma_point_scheme == 3):
            sigma_points.data.copy_(kalman.calc_sigma_points_p3(mu_uks[1:], sqrt_dimZ_p_lamb, kalman.cholesky_save(var_uks[1:])).data)
        elif (sigma_point_scheme == 5):
            sigma_points.data.copy_(kalman.calc_sigma_points_p5(mu_uks[1:], sqrt_dimZ_p_lamb, kalman.cholesky_save(var_uks[1:])).data)
        elif (sigma_point_scheme == 0):
            sigma_points.data.copy_(kalman.calc_sigma_points_rand(mu_uks[1:], sqrt_dimZ_p_lamb, kalman.cholesky_save(var_uks[1:]),
                                                                  nSigmaPoints).data)
        sigma_points_g.data.copy_(model.fcn_emission_free(sigma_points.reshape(nT * nSigmaPoints, dim_z),
                                                          args_model)[:, measure_mask_exclude].reshape(nT, nSigmaPoints, dim_x).data)
        # pairwise sigma points
        outer_z.data.copy_(torch.einsum('tij,tjk->tik', (G_uks, var_uks[1:])).data)
        var_pairwise.data.copy_(torch.cat([torch.cat([var_uks[1:], outer_z.transpose(2, 1)], 2),
                                           torch.cat([outer_z, var_uks[:-1]], 2)], 1).data)
        mu_pairwise.data.copy_(torch.cat([mu_uks[1:], mu_uks[:-1]], 1).data)
        if (sigma_point_scheme == 3):
            sigma_points_pairwise.data.copy_(kalman.calc_sigma_points_p3(mu_pairwise, sqrt_dimZ_p_lamb2, kalman.cholesky_save(var_pairwise)).data)
        elif (sigma_point_scheme == 5):
            sigma_points_pairwise.data.copy_(kalman.calc_sigma_points_p5(mu_pairwise, sqrt_dimZ_p_lamb2, kalman.cholesky_save(var_pairwise)).data)
        elif (sigma_point_scheme == 0):
            sigma_points_pairwise.data.copy_(kalman.calc_sigma_points_rand(mu_pairwise, sqrt_dimZ_p_lamb2, kalman.cholesky_save(var_pairwise),
                                                                           nSigmaPoints2).data)
        # WHEN A IS NOT LEARNED
        # var_f
        x1_m_fx0.data.copy_(kalman.substraction_save(sigma_points_pairwise[:, :, :dim_z],
                                                     sigma_points_pairwise[:, :, dim_z:]).data)
        var_f_1.data.copy_(torch.sum((w_c2[None, :, None, None] / float(nT)) * \
                                     torch.einsum('tmi,tmj->tmij',
                                                  (x1_m_fx0, x1_m_fx0)), (0, 1)).data)
        var_f_1.data.copy_((0.5 * (var_f_1 + var_f_1.transpose(1, 0))).data)
        
        # var_g
        y1_m_hx1_2.data.copy_(torch.sum(w_c[None, :, None] * \
                                        kalman.substraction_save(measure[:, None, :],
                                                                 sigma_points_g)**2, 1).data)
        y1_m_hx1_2[~measure_mask] = 0.0
        #
        measurement_expectation.data.copy_(sigma_points_g[:, 0, :].data)
        #
        var_g_1_L.data.copy_((torch.sum(y1_m_hx1_2, 0) * nMeasureT**-1).data)
    return

# IMPROVE: put arrays of args_Qg into args
def run(mu0_in, var0_in, A_in, var_f_in, var_g_in,
        args):
    tol_out = float(cfg.tol)
    iter_out_max = int(cfg.iter_max)
    #
    nSave = int(50)
    #
    verbose = bool(True)

    use_cuda = args['use_cuda']
    args_kalman = args['args_kalman']
    nT = args_kalman['nT']
    dim_z = args_kalman['dim_z']
    dim_x = args_kalman['dim_x']
    args_Qg = args_kalman['args_Qg']
    #
    x1_m_fx0 = args_kalman['x1_m_fx0']
    y1_m_hx1_2 = args_kalman['y1_m_hx1_2']
    
    # theta0
    mu0_0 = mu0_in.clone()
    var0_0 = var0_in.clone()
    A_0 = A_in.clone()
    var_f_0 = var_f_in.clone()
    var_g_0 = var_g_in.clone()
    var_g_0_L = torch.diag(var_g_0).clone()
    # theta1
    mu0_1 = mu0_in.clone()
    var0_1 = var0_in.clone()
    A_1 = A_in.clone()
    var_f_1 = var_f_in.clone()
    var_g_1 = var_g_in.clone()
    var_g_1_L = torch.diag(var_g_0).clone()
    #
    measurement_expectation0 = args_kalman['measurement_expectation0']
    measurement_expectation1 = args_kalman['measurement_expectation1']
    # dTheta
    d_mu0 = mu0_in.clone()
    d_var0 = var0_in.clone()
    d_A = A_in.clone()
    d_var_f = var_f_in.clone()
    d_var_g = var_g_in.clone()
    d_norm_mu0 = mu0_in.clone()
    d_norm_var0 = var0_in.clone()
    d_norm_A = A_in.clone()
    d_norm_var_f = var_f_in.clone()
    d_norm_var_g = var_g_in.clone()
    #
    iter_out = int(0)
    cond = bool(False)
    convergence_status = int(0)
    while not(cond):        
        # E-STEP
        mu_kalman, var_kalman, G_kalman = kalman.uks(mu0_0, var0_0, A_0, var_f_0, var_g_0,
                                                     args)   
        # M-STEP
        # mu0
        mu0_1.data.copy_(mu_kalman[0].data)
        # var0
        var0_1.data.copy_(var_kalman[0].data)
        # var_f & var_g & A
        update_theta(mu_kalman, var_kalman, G_kalman,
                     args, args_Qg,
                     x1_m_fx0, y1_m_hx1_2, measurement_expectation1,
                     var_f_1, var_g_1_L, A_1)
        var_g_1.diagonal().data.copy_(var_g_1_L.data) # since var_g is assumed to be a diagonal matrix
        # REST
        # dTheata
        d_mu0.data.copy_(abs((mu0_1 - mu0_0)).data)
        d_var0.data.copy_(abs((var0_1 - var0_0)).data)
        d_A.data.copy_(abs((A_1 - A_0)).data)
        d_var_f.data.copy_(abs((var_f_1 - var_f_0)).data)
        d_var_g.data.copy_(abs((var_g_1 - var_g_0)).data)
        d_norm_mu0.data.copy_((d_mu0 / abs(mu0_0)).data)
        d_norm_var0.data.copy_((d_var0 / abs(var0_0)).data)
        d_norm_A.data.copy_((d_A / abs(A_0)).data)
        d_norm_var_f.data.copy_((d_var_f / abs(var_f_0)).data)
        d_norm_var_g.data.copy_((d_var_g / abs(var_g_0)).data)
        # differences in theta
        d_norm_mu0_mean = float(torch.mean(d_norm_mu0)) # dim_z
        d_norm_var0_mean = float(torch.mean(d_norm_var0.diag())) # dim_z
        d_norm_var_f_mean = float(torch.mean(d_norm_var_f.diag())) # dim_z
        d_norm_var_g_mean = float(torch.mean(d_norm_var_g.diag())) # dim_x
        theta_mean = float(torch.mean(torch.cat([d_norm_mu0,
                                                 d_norm_var0.diag(),
                                                 d_norm_var_f.diag(),
                                                 d_norm_var_g.diag()], 0)))
        #
        if (theta_mean < tol_out):
            cond = True
            verbose = True
            convergence_status = 1
        if (iter_out >= iter_out_max):
            cond = True
            verbose = True
            convergence_status = 2
                
        # PRINT
        if (verbose):
            print('iteration:\t\t{:07d} / {:07d}'.format(iter_out, iter_out_max))
            print('min./max. theta0:')
            print('min./max. mu0_t0:\t{:0.8e} / {:0.8e}'.format(float(torch.min(mu0_0[:3])), float(torch.max(mu0_0[:3]))))
            print('min./max. mu0_r0:\t{:0.8e} / {:0.8e}'.format(float(torch.min(mu0_0[3:6])), float(torch.max(mu0_0[3:6]))))
            print('min./max. mu0_r:\t{:0.8e} / {:0.8e}'.format(float(torch.min(mu0_0[6:])), float(torch.max(mu0_0[6:]))))
            print('var0:\t\t\t{:0.8e} / {:0.8e}'.format(float(torch.min(var0_0)),
                                                        float(torch.max(var0_0))))
            print('var0_diag:\t\t{:0.8e} / {:0.8e}'.format(float(torch.min(torch.diag(var0_0))),
                                                           float(torch.max(torch.diag(var0_0)))))
            print('A:\t\t\t{:0.8e} / {:0.8e}'.format(float(torch.min(A_0)),
                                                     float(torch.max(A_0))))
            print('var_f:\t\t\t{:0.8e} / {:0.8e}'.format(float(torch.min(var_f_0)),
                                                         float(torch.max(var_f_0))))
            print('var_f_diag:\t\t{:0.8e} / {:0.8e}'.format(float(torch.min(torch.diag(var_f_0))),
                                                            float(torch.max(torch.diag(var_f_0)))))
            print('var_g_diag:\t\t{:0.8e} / {:0.8e}'.format(float(torch.min(torch.diag(var_g_0))),
                                                            float(torch.max(torch.diag(var_g_0)))))
            print('log det var0:\t\t{:0.8e}'.format(float(torch.logdet(var0_0))))
            print('log det var_f:\t\t{:0.8e}'.format(float(torch.logdet(var_f_0))))
            print('log det var_g:\t\t{:0.8e}'.format(float(torch.logdet(var_g_0))))
            print('min./max. theta1:')
            print('min./max. mu0_t0:\t{:0.8e} / {:0.8e}'.format(float(torch.min(mu0_1[:3])), float(torch.max(mu0_1[:3]))))
            print('min./max. mu0_r0:\t{:0.8e} / {:0.8e}'.format(float(torch.min(mu0_1[3:6])), float(torch.max(mu0_1[3:6]))))
            print('min./max. mu0_r:\t{:0.8e} / {:0.8e}'.format(float(torch.min(mu0_1[6:])), float(torch.max(mu0_1[6:]))))
            print('var0:\t\t\t{:0.8e} / {:0.8e}'.format(float(torch.min(var0_1)),
                                                        float(torch.max(var0_1))))
            print('var0_diag:\t\t{:0.8e} / {:0.8e}'.format(float(torch.min(torch.diag(var0_1))),
                                                           float(torch.max(torch.diag(var0_1)))))
            print('A:\t\t\t{:0.8e} / {:0.8e}'.format(float(torch.min(A_1)),
                                                     float(torch.max(A_1))))
            print('var_f:\t\t\t{:0.8e} / {:0.8e}'.format(float(torch.min(var_f_1)),
                                                         float(torch.max(var_f_1))))
            print('var_f_diag:\t\t{:0.8e} / {:0.8e}'.format(float(torch.min(torch.diag(var_f_1))),
                                                            float(torch.max(torch.diag(var_f_1)))))
            print('var_g_diag:\t\t{:0.8e} / {:0.8e}'.format(float(torch.min(torch.diag(var_g_1))),
                                                            float(torch.max(torch.diag(var_g_1)))))
            print('log det var0:\t\t{:0.8e}'.format(float(torch.logdet(var0_1))))
            print('log det var_f:\t\t{:0.8e}'.format(float(torch.logdet(var_f_1))))
            print('log det var_g:\t\t{:0.8e}'.format(float(torch.logdet(var_g_1))))
            print('|theta1 - theta0|:')
            print('mu0_t0:\t\t\t{:0.8e} / {:0.8e} / {:0.8e}\t{:0.8e}'.format(float(torch.min(d_mu0[:3])),
                                                                             float(torch.mean(d_mu0[:3])),
                                                                             float(torch.max(d_mu0[:3])),
                                                                             float(torch.median(d_mu0[:3]))))
            print('mu0_r0:\t\t\t{:0.8e} / {:0.8e} / {:0.8e}\t{:0.8e}'.format(float(torch.min(d_mu0[3:6])),
                                                                             float(torch.mean(d_mu0[3:6])),
                                                                             float(torch.max(d_mu0[3:6])),
                                                                             float(torch.median(d_mu0[3:6]))))
            print('mu0_r:\t\t\t{:0.8e} / {:0.8e} / {:0.8e}\t{:0.8e}'.format(float(torch.min(d_mu0[6:])),
                                                                            float(torch.mean(d_mu0[6:])),
                                                                            float(torch.max(d_mu0[6:])),
                                                                            float(torch.median(d_mu0[6:]))))
            print('var0:\t\t\t{:0.8e} / {:0.8e} / {:0.8e}\t{:0.8e}'.format(float(torch.min(d_var0[~torch.isnan(d_var0)])),
                                                                           float(torch.mean(d_var0[~torch.isnan(d_var0)])),
                                                                           float(torch.max(d_var0[~torch.isnan(d_var0)])),
                                                                           float(torch.median(d_var0[~torch.isnan(d_var0)]))))
            print('var0_diag:\t\t{:0.8e} / {:0.8e} / {:0.8e}\t{:0.8e}'.format(float(torch.min(torch.diag(d_var0))),
                                                                              float(torch.mean(torch.diag(d_var0))),
                                                                              float(torch.max(torch.diag(d_var0))),
                                                                              float(torch.median(torch.diag(d_var0)))))
            print('A:\t\t\t{:0.8e} / {:0.8e} / {:0.8e}\t{:0.8e}'.format(float(torch.min(d_A[~torch.isnan(d_A)])),
                                                                        float(torch.mean(d_A[~torch.isnan(d_A)])),
                                                                        float(torch.max(d_A[~torch.isnan(d_A)])),
                                                                        float(torch.median(d_A[~torch.isnan(d_A)]))))
            print('var_f:\t\t\t{:0.8e} / {:0.8e} / {:0.8e}\t{:0.8e}'.format(float(torch.min(d_var_f[~torch.isnan(d_var_f)])),
                                                                            float(torch.mean(d_var_f[~torch.isnan(d_var_f)])),
                                                                            float(torch.max(d_var_f[~torch.isnan(d_var_f)])),
                                                                            float(torch.median(d_var_f[~torch.isnan(d_var_f)]))))
            print('var_f_diag:\t\t{:0.8e} / {:0.8e} / {:0.8e}\t{:0.8e}'.format(float(torch.min(torch.diag(d_var_f))),
                                                                               float(torch.mean(torch.diag(d_var_f))),
                                                                               float(torch.max(torch.diag(d_var_f))),
                                                                               float(torch.median(torch.diag(d_var_f)))))
            print('var_g_diag:\t\t{:0.8e} / {:0.8e} / {:0.8e}\t{:0.8e}'.format(float(torch.min(torch.diag(d_var_g))),
                                                                               float(torch.mean(torch.diag(d_var_g))),
                                                                               float(torch.max(torch.diag(d_var_g))),
                                                                               float(torch.median(torch.diag(d_var_g)))))
            print('norm. |theta1 - theta0|:')
            print('mu0_t0:\t\t\t{:0.8e} / {:0.8e} / {:0.8e}\t{:0.8e}'.format(float(torch.min(d_norm_mu0[:3])),
                                                                             float(torch.mean(d_norm_mu0[:3])),
                                                                             float(torch.max(d_norm_mu0[:3])),
                                                                             float(torch.median(d_norm_mu0[:3]))))
            print('mu0_r0:\t\t\t{:0.8e} / {:0.8e} / {:0.8e}\t{:0.8e}'.format(float(torch.min(d_norm_mu0[3:6])),
                                                                             float(torch.mean(d_norm_mu0[3:6])),
                                                                             float(torch.max(d_norm_mu0[3:6])),
                                                                             float(torch.median(d_norm_mu0[3:6]))))
            print('mu0_r:\t\t\t{:0.8e} / {:0.8e} / {:0.8e}\t{:0.8e}'.format(float(torch.min(d_norm_mu0[6:])),
                                                                            float(torch.mean(d_norm_mu0[6:])),
                                                                            float(torch.max(d_norm_mu0[6:])),
                                                                            float(torch.median(d_norm_mu0[6:]))))
            print('var0:\t\t\t{:0.8e} / {:0.8e} / {:0.8e}\t{:0.8e}'.format(float(torch.min(d_norm_var0[~torch.isnan(d_norm_var0)])),
                                                                           float(torch.mean(d_norm_var0[~torch.isnan(d_norm_var0)])),
                                                                           float(torch.max(d_norm_var0[~torch.isnan(d_norm_var0)])),
                                                                           float(torch.median(d_norm_var0[~torch.isnan(d_norm_var0)]))))
            print('var0_diag:\t\t{:0.8e} / {:0.8e} / {:0.8e}\t{:0.8e}'.format(float(torch.min(torch.diag(d_norm_var0))),
                                                                              float(torch.mean(torch.diag(d_norm_var0))),
                                                                              float(torch.max(torch.diag(d_norm_var0))),
                                                                              float(torch.median(torch.diag(d_norm_var0)))))
            print('A:\t\t\t{:0.8e} / {:0.8e} / {:0.8e}\t{:0.8e}'.format(float(torch.min(d_norm_A[~torch.isnan(d_norm_A)])),
                                                                        float(torch.mean(d_norm_A[~torch.isnan(d_norm_A)])),
                                                                        float(torch.max(d_norm_A[~torch.isnan(d_norm_A)])),
                                                                        float(torch.median(d_norm_A[~torch.isnan(d_norm_A)]))))
            print('var_f:\t\t\t{:0.8e} / {:0.8e} / {:0.8e}\t{:0.8e}'.format(float(torch.min(d_norm_var_f[~torch.isnan(d_norm_var_f)])),
                                                                            float(torch.mean(d_norm_var_f[~torch.isnan(d_norm_var_f)])),
                                                                            float(torch.max(d_norm_var_f[~torch.isnan(d_norm_var_f)])),
                                                                            float(torch.median(d_norm_var_f[~torch.isnan(d_norm_var_f)]))))
            print('var_f_diag:\t\t{:0.8e} / {:0.8e} / {:0.8e}\t{:0.8e}'.format(float(torch.min(torch.diag(d_norm_var_f))),
                                                                               float(torch.mean(torch.diag(d_norm_var_f))),
                                                                               float(torch.max(torch.diag(d_norm_var_f))),
                                                                               float(torch.median(torch.diag(d_norm_var_f)))))
            print('var_g_diag:\t\t{:0.8e} / {:0.8e} / {:0.8e}\t{:0.8e}'.format(float(torch.min(torch.diag(d_norm_var_g))),
                                                                               float(torch.mean(torch.diag(d_norm_var_g))),
                                                                               float(torch.max(torch.diag(d_norm_var_g))),
                                                                               float(torch.median(torch.diag(d_norm_var_g)))))
            print('summary statistics:')
            print('E(norm. |mu0_1 - mu0_0|):\t{:0.8e}'.format(d_norm_mu0_mean))
            print('E(norm. |var0_1 - var0_0|):\t{:0.8e}'.format(d_norm_var0_mean))
            print('E(norm. |var_f_1 - var_f_0|):\t{:0.8e}'.format(d_norm_var_f_mean))
            print('E(norm. |var_g_1 - var_g_0|):\t{:0.8e}'.format(d_norm_var_g_mean))
            print('E(norm. |theta_1 - theta_0|):\t{:0.8e} (goal: {:0.8e})'.format(theta_mean, tol_out))
            print()
            #
            if (cond):
                if (convergence_status == 1):
                    print('CONVERGENCE')
                elif (convergence_status == 2):
                    print('MAX. NUMBER OF ITERATIONS REACHED')
                print('FINISHED POSE RECONSTRUCTION ({:s})'.format(cfg.folder_save))
                print()

        # update arrays for next iteration
        if not(cond):
            iter_out += 1
            # parameters
            mu0_0.data.copy_(mu0_1.data)
            var0_0.data.copy_(var0_1.data)
            A_0.data.copy_(A_1.data)
            var_f_0.data.copy_(var_f_1.data)
            var_g_0.data.copy_(var_g_1.data)
            var_g_0_L.data.copy_(var_g_1_L.data)
            # other
            measurement_expectation0.data.copy_(measurement_expectation1.data)

        # save (change this to torch.save)
        if (((iter_out % nSave) == 0) or cond):
            save_dict = dict()
            if (cond):
                if (convergence_status == 1):
                    save_dict['status'] = 1
                    save_dict['message'] = 'CONVERGENCE'
                elif (convergence_status == 2):
                    save_dict['status'] = 2
                    save_dict['message'] = 'MAX. NUMBER OF ITERATIONS REACHED'
            #
            save_dict['mode'] = cfg.mode
            save_dict['mu0'] = mu0_1.detach().cpu().numpy() 
            save_dict['var0'] = var0_1.detach().cpu().numpy() 
            save_dict['A'] = A_1.detach().cpu().numpy() 
            save_dict['var_f'] = var_f_1.detach().cpu().numpy() 
            save_dict['var_g'] = var_g_1.detach().cpu().numpy()
            #
            save_dict['mu_uks'] = mu_kalman.detach().cpu().numpy() 
            save_dict['var_uks'] = var_kalman.detach().cpu().numpy()            
            #
            save_dict['cond'] = cond
            save_dict['iter_out'] = iter_out
            #
            np.save(os.path.join(cfg.folder_save,'save_dict.npy'), save_dict)
    return mu0_1, var0_1, A_1, var_f_1, var_g_1
