#!/usr/bin/env python3

import torch

from . import model

def addition_save(term1, term2):
    return term1 + term2

def substraction_save(term1, term2):
    return term1 - term2

def cholesky_save(M_in):
    len_MShape = len(M_in.size())
    if (len_MShape == 2): # no batch
        try:
            L = torch.cholesky(0.5 * (M_in + M_in.transpose(1, 0)))
        except:
            eps = torch.diag_embed(torch.ones_like(M_in[0], dtype=model.float_type))
            expo = -52
            cholesky_is_invalid = True
            while (cholesky_is_invalid):
                try:
                    L = torch.cholesky(0.5 * (M_in + M_in.transpose(1, 0)) + eps * 2.0**expo)
                    cholesky_is_invalid = False
                except:
                    expo += 1
            print('WARNING: cholesky_save (no batch) expo:\t{:03d}'.format(int(expo)))
    elif (len_MShape == 3): # batch
        try:
            L = torch.cholesky(0.5 * (M_in + M_in.transpose(2, 1)))
        except:
            eps = torch.diag_embed(torch.ones_like(M_in[0, 0], dtype=model.float_type))
            expo = -52
            cholesky_is_invalid = True
            while (cholesky_is_invalid):
                try:
                    L = torch.cholesky(0.5 * (M_in + M_in.transpose(2, 1)) + eps[None, :, :] * 2.0**expo)
                    cholesky_is_invalid = False
                except:
                    expo += 1
            print('WARNING: cholesky_save (batch) expo:\t{:03d}'.format(int(expo)))
    else:
        print('ERROR: Invalid array dimension in kalman.cholesky_save')
    return L


def calc_sigma_points_rand(m, w, L,
                           n):
    distribution = torch.distributions.MultivariateNormal(loc=torch.zeros_like(m, dtype=model.float_type),
                                                          scale_tril=L)
    
    len_mShape = len(m.size())
    nSamples = int((n-1)/2)
    if (len_mShape == 1):
        samples = distribution.sample((nSamples,))
        sigma_points_rand = torch.cat([m[None, :],
                                       addition_save(m[None, :], samples),
                                       addition_save(m[None, :], -samples)], 0)
    elif (len_mShape == 2):
        samples = distribution.sample((nSamples,)).permute(1, 0, 2)           
        sigma_points_rand = torch.cat([m[:, None, :],
                                       kalma.addition_save(m[:, None, :], samples),
                                       kalma.addition_save(m[:, None, :], -samples)], 1)
    else:
        print('ERROR: Invalid array dimension in kalman.calc_sigma_points_rand')
        raise
    return sigma_points_rand

def calc_sigma_points_p3(m, w, L):
    len_mShape = len(m.size())
    if (len_mShape == 1):
        w_times_L_T = w * L.transpose(1, 0)
        sigma_points_p3 = torch.cat([m[None, :],
                                     addition_save(m[None, :], w_times_L_T),
                                     addition_save(m[None, :], -w_times_L_T)], 0)
    elif (len_mShape == 2):
        w_times_L_T = w * L.transpose(2, 1)
        sigma_points_p3 = torch.cat([m[:, None, :],
                                     addition_save(m[:, None, :], w_times_L_T),
                                     addition_save(m[:, None, :], -w_times_L_T)], 1)
    else:
        print('ERROR: Invalid array dimension in kalman.calc_sigma_points_p3')
        raise
    return sigma_points_p3

def calc_sigma_points_p5(m, w, L):
    # 3
    sigma_points_p3 = calc_sigma_points_p3(m, w, L)
    
    # 5
    mShape = list(m.size())
    if (len(mShape) == 1):
        dim_z = mShape[0]  
        dim0 = (dim_z*(dim_z-1))
        
        L_T = L.transpose(1, 0)

        # can be calculated outside already
        L5_mask = torch.ones((dim_z, dim_z, dim_z), dtype=torch.bool).cuda()
        L5_mask_index = torch.arange(dim_z, dtype=torch.int64).cuda()
        L5_mask[L5_mask_index, L5_mask_index, :] = 0
        
        L5_1 = addition_save(L_T[None, :, :], L_T[:, None, :])[L5_mask].reshape(dim0, dim_z)
        L5_2 = addition_save(L_T[None, :, :], -L_T[:, None, :])[L5_mask].reshape(dim0, dim_z)
        L5_3 = addition_save(-L_T[None, :, :], L_T[:, None, :])[L5_mask].reshape(dim0, dim_z)
        L5_4 = addition_save(-L_T[None, :, :], -L_T[:, None, :])[L5_mask].reshape(dim0, dim_z)
        
        L5 = torch.unique(torch.cat([L5_1, L5_2, L5_3, L5_4], 0).float(),
                          dim=0).double()

        sigma_points_p5 = torch.cat([sigma_points_p3,
                                     addition_save(m[None, :], w * L5)], 0)
    elif (len(mShape) == 2):
        nT = mShape[0]
        dim_z = mShape[1]

        L_T = L.transpose(2, 1)
        
        L1 = L_T.repeat(dim_z, 1, 1, 1)
        L1 = L1.reshape(dim_z, nT, dim_z, dim_z)
        L1 = L1.transpose(1, 0)
        L2 = L_T.repeat(1, 1, 1, dim_z)
        L2 = L2.reshape(nT, dim_z, dim_z, dim_z)
        
        L5_1 = addition_save(L1, L2)
        L5_2 = addition_save(L1, -L2)
        L5_3 = addition_save(-L1, L2)
        L5_4 = addition_save(-L1, -L2)

        # can be calculated outside already
        L5_mask = torch.ones_like(L1[0], dtype=torch.bool)
        L5_mask_mask = torch.ones_like(L_T[0, 0], dtype=torch.bool)
        L5_mask_mask = torch.diag_embed(L5_mask_mask)
        L5_mask_mask = L5_mask_mask.repeat(1, 1, dim_z)
        L5_mask_mask = L5_mask_mask.reshape(dim_z, dim_z, dim_z)
        L5_mask_mask = L5_mask_mask.transpose(2, 1)
        L5_mask[L5_mask_mask] = 0
        L5_mask = L5_mask.repeat(nT, 1, 1, 1)
        L5_mask = L5_mask.reshape(nT, dim_z, dim_z, dim_z)
        
        dim0 = (dim_z*(dim_z-1))
        
        L5_1 = L5_1[L5_mask].reshape(nT, dim0, dim_z)
        L5_2 = L5_2[L5_mask].reshape(nT, dim0, dim_z)
        L5_3 = L5_3[L5_mask].reshape(nT, dim0, dim_z)
        L5_4 = L5_4[L5_mask].reshape(nT, dim0, dim_z)
        L5 = torch.cat([L5_1, L5_2, L5_3, L5_4], 1)
        
        L5 = torch.unique(L5.float(), dim=1).double()
        
        nSigmaPointsAdd = 2*dim_z*(dim_z-1)
        m_use = m.repeat(1, 1, nSigmaPointsAdd)
        m_use = m_use.reshape(nT, nSigmaPointsAdd, dim_z)

        sigma_points_p5 = torch.cat([sigma_points_p3,
                                     m_use + w * L5], 1)
    else:
        print('ERROR: Invalid array dimension in kalman.calc_sigma_points_p5')
        raise
    return sigma_points_p5

def uks(mu0, var0, A, var_f, var_g,
        args):
    nT = args['args_kalman']['nT']
    G_kalman = args['args_kalman']['G_kalman']

    mu_kalman, var_kalman = ukf(mu0, var0, A, var_f, var_g,
                                args)

    # should not be necessary since arrays will be overwritten anyway 
    G_kalman.data.zero_() 

    for t in range(nT-1, -1, -1):
        uks_step(mu_kalman[t], var_kalman[t], G_kalman[t],
                 mu_kalman[t+1], var_kalman[t+1],
                 A, var_f, var_g,
                 args)
    return mu_kalman, var_kalman, G_kalman

# 2013__Saerkkae__Bayesian_Filtering_And_Smoothing
# page 148f., algorithm 9.3
def uks_step(mu_ukf_t, var_ukf_t, G_uks,
             mu_uks_tp1, var_uks_tp1,
             A, var_f, var_g,
             args):
    dim_z = args['args_kalman']['dim_z']
    sigma_point_scheme = args['args_kalman']['sigma_point_scheme']
    w_m = args['args_kalman']['w_m']
    w_c = args['args_kalman']['w_c']
    sqrt_dimZ_p_lamb = args['args_kalman']['sqrt_dimZ_p_lamb']
    args_model = args['args_model']  
    
    kalman_sigmaPoints = args['args_kalman']['sigmaPoints_kalman']
    var_2 = args['args_kalman']['var_2_uks']
    var_31 = args['args_kalman']['var_31_uks']
    var_32 = args['args_kalman']['var_32_uks']
    var_33 = args['args_kalman']['var_33_uks']

    # should not be necessary since arrays will be overwritten anyway 
    var_2.data.zero_()
    var_31.data.zero_() 
    var_32.data.zero_() 
    var_33.data.zero_() 
    
    # variable shapes:
    # var_1:  nSigmaPoints, dim_z
    # var_2:  nSigmaPoints, dim_z
    # var_31: dim_z
    # var_32: dim_z, dim_z
    # var_33: dim_z, dim_z
    # var_41: dim_z, dim_z (G)
    # var_42: dim_z (mu)
    # var_43: dim_z, dim_z (var)

    # 1.
    if (sigma_point_scheme == 3):
        kalman_sigmaPoints.data.copy_(calc_sigma_points_p3(mu_ukf_t, sqrt_dimZ_p_lamb, cholesky_save(var_ukf_t)).data)
    elif (sigma_point_scheme == 5):
        kalman_sigmaPoints.data.copy_(calc_sigma_points_p5(mu_ukf_t, sqrt_dimZ_p_lamb, cholesky_save(var_ukf_t)).data)
    elif (sigma_point_scheme == 0):
        kalman_sigmaPoints.data.copy_(calc_sigma_points_rand(mu_ukf_t, sqrt_dimZ_p_lamb, cholesky_save(var_ukf_t),
                                                             args['args_kalman']['nSigmaPoints']).data)
        
    # 2.
    var_2.data.copy_(model.fcn_transition_free(kalman_sigmaPoints, A, args_model).data)

    # 3.1
    var_31.data.copy_(torch.sum(w_m[:, None] * var_2, 0).data)

    # var_2 is treated as dummy to save intermediate calculations
    var_2.data.copy_(substraction_save(var_2, var_31).data)
    
    # 3.2
    var_32.data.copy_(addition_save(torch.sum(w_c[:, None, None] * \
                                              torch.einsum('ni,nj->nij', (var_2, var_2)), 0),
                                    var_f).data)
    var_32.data.copy_((0.5 * (var_32 + var_32.transpose(1, 0))).data)
    
    # 3.3 [cross-covariance -> not symmetric]
    var_33.data.copy_(torch.sum(w_c[:, None, None] * \
                                torch.einsum('ni,nj->nij',
                                             (substraction_save(kalman_sigmaPoints, mu_ukf_t),
                                              var_2)), 0).data)
    
    # 4.1
    G_uks.data.copy_(torch.mm(var_33, torch.inverse(var_32)).data)    
    
    # 4.2
    mu_ukf_t.data.add_(torch.mv(G_uks,
                                substraction_save(mu_uks_tp1, var_31)).data)    
    
    # 4.3 [P + G (P_s - P_m) G_T = P + (G P_s - D) G_T since G = D P_m^-1]
    var_ukf_t.data.add_(torch.mm(substraction_save(torch.mm(G_uks, var_uks_tp1), var_33),
                                 G_uks.transpose(1, 0)).data)
    var_ukf_t.data.copy_((0.5 * (var_ukf_t + var_ukf_t.transpose(1, 0))).data)
    return
    
def ukf(mu0, var0, A, var_f, var_g,
        args):
    nT = args['args_kalman']['nT']
    dim_z = args['args_kalman']['dim_z']
    measure = args['args_kalman']['measure']
    measure_mask = args['args_kalman']['measure_mask']
    measure_mask_exclude = args['args_kalman']['measure_mask_exclude']

    mu_kalman = args['args_kalman']['mu_kalman'] 
    var_kalman = args['args_kalman']['var_kalman']
    
    # should not be necessary since arrays will be overwritten anyway  
    mu_kalman.data.zero_()
    var_kalman.data.zero_()
    
    mu_kalman[0].data.copy_(mu0.data)
    var_kalman[0].data.copy_(var0.data)
    for t in range(nT):
        ukf_step(mu_kalman[t], var_kalman[t],
                 measure[t],
                 measure_mask[t],
                 measure_mask_exclude,
                 A, var_f, var_g,
                 args,
                 mu_kalman[t+1], var_kalman[t+1])
    return mu_kalman, var_kalman

# 2013__Saerkkae__Bayesian_Filtering_And_Smoothing
# page 86f., algorithm 5.14
# 2016__Shumway__Time_Series_Analysis_and_its_Applications
# page 310f., 6.4. Missing Data Modifications
def ukf_step(mu_t, var_t,
             x_t,
             x_t_mask,
             x_t_mask_exclude,
             A, var_f, var_g,
             args,
             mu_ukf, var_ukf):
    dim_z = args['args_kalman']['dim_z']
    sigma_point_scheme = args['args_kalman']['sigma_point_scheme']
    w_m = args['args_kalman']['w_m']
    w_c = args['args_kalman']['w_c']
    sqrt_dimZ_p_lamb = args['args_kalman']['sqrt_dimZ_p_lamb']
    args_model = args['args_model']
    
    kalman_sigmaPoints = args['args_kalman']['sigmaPoints_kalman']
    var_p2 = args['args_kalman']['var_p2_ukf']
    var_p31 = args['args_kalman']['var_p31_ukf']
    var_p32 = args['args_kalman']['var_p32_ukf']
    var_u2 = args['args_kalman']['var_u2_ukf']
    var_u31 = args['args_kalman']['var_u31_ukf']
    var_u32 = args['args_kalman']['var_u32_ukf']
    var_u33 = args['args_kalman']['var_u33_ukf']
    var_u41 = args['args_kalman']['var_u41_ukf']
    
    # should not be necessary since arrays will be overwritten anyway 
    var_p2.data.zero_()
    var_p31.data.zero_() 
    var_p32.data.zero_() 
    var_u2.data.zero_() 
    var_u31.data.zero_()
    var_u32.data.zero_() 
    var_u33.data.zero_() 
    var_u41.data.zero_() 
    
    # variable shapes:
    # var_p1:  nSigmaPoints, dim_z
    # var_p2:  nSigmaPoints, dim_z
    # var_p31: dim_z
    # var_p32: dim_z, dim_z
    # var_u1:  nSigmaPoints, dim_z
    # var_u2:  nSigmaPoints, dim_x
    # var_u31: dim_x
    # var_u32: dim_x, dim_x       
    # var_u33: dim_z, dim_x     
    # var_u41: dim_z, dim_x
    # var_u42: dim_z (mu)
    # var_u43: dim_z, dim_z (var)
    
    # PREDICT
    # p.1
    if (sigma_point_scheme == 3):
        kalman_sigmaPoints.data.copy_(calc_sigma_points_p3(mu_t, sqrt_dimZ_p_lamb, cholesky_save(var_t)).data)
    elif (sigma_point_scheme == 5):
        kalman_sigmaPoints.data.copy_(calc_sigma_points_p5(mu_t, sqrt_dimZ_p_lamb, cholesky_save(var_t)).data)
    elif (sigma_point_scheme == 0):
        kalman_sigmaPoints.data.copy_(calc_sigma_points_rand(mu_t, sqrt_dimZ_p_lamb, cholesky_save(var_t),
                                                             args['args_kalman']['nSigmaPoints']).data)
        
    # p.2
    var_p2.data.copy_(model.fcn_transition_free(kalman_sigmaPoints, A, args_model).data)

    # p.3.1
    var_p31.data.copy_(torch.sum(w_m[:, None] * var_p2, 0).data)
    
    # var_p2 is treated as dummy to save intermediate calculations
    var_p2.data.copy_(substraction_save(var_p2, var_p31).data)
    
    # p.3.2
    var_p32.data.copy_(addition_save(torch.sum(w_c[:, None, None] * \
                                               torch.einsum('ni,nj->nij', (var_p2, var_p2)), 0),
                                     var_f).data)
    var_p32.data.copy_((0.5 * (var_p32 + var_p32.transpose(1, 0))).data)

    # UPDATE
    # u.1
    if (sigma_point_scheme == 3):
        kalman_sigmaPoints.data.copy_(calc_sigma_points_p3(var_p31, sqrt_dimZ_p_lamb, cholesky_save(var_p32)).data)
    elif (sigma_point_scheme == 5):
        kalman_sigmaPoints.data.copy_(calc_sigma_points_p5(var_p31, sqrt_dimZ_p_lamb, cholesky_save(var_p32)).data)
    elif (sigma_point_scheme == 0):
        kalman_sigmaPoints.data.copy_(calc_sigma_points_rand(var_p31, sqrt_dimZ_p_lamb, cholesky_save(var_p32),
                                                             args['args_kalman']['nSigmaPoints']).data)
        
    # u.2
    var_u2.data.copy_(model.fcn_emission_free(kalman_sigmaPoints, args_model)[:, x_t_mask_exclude].data)    
    
    # u.3.1
    var_u31.data.copy_(torch.sum(w_m[:, None] * var_u2, 0).data)
    
    # var_u2 is treated as dummy to save intermediate calculations
    var_u2.data.copy_(substraction_save(var_u2, var_u31).data)
    
    # u.3.2 (i.e. S)
    var_u32.data.copy_(addition_save(torch.sum(w_c[:, None, None] * \
                                               torch.einsum('ni,nj->nij', (var_u2, var_u2)), 0),
                                     var_g).data)
    var_u32[~x_t_mask, :] = 0.0 # according to 6.24 (sigma used in 6.22) in Shumway et al. (i.e. after substition of A, c.f. 6.77 & 6.78)
    var_u32[:, ~x_t_mask] = 0.0 # according to 6.24 (sigma used in 6.22) in Shumway et al. (i.e. after substition of A, c.f. 6.77 & 6.78)
    var_u32[~x_t_mask, ~x_t_mask] = 1.0 # according to 6.24 (sigma used in 6.22) in Shumway et al. (i.e. after substition of A, c.f. 6.77 & 6.78)
    var_u32.data.copy_((0.5 * (var_u32 + var_u32.transpose(1, 0))).data)
    
    # u.3.3 (i.e. C) [cross-covariance -> not symmetric]
    var_u33.data.copy_(torch.sum(w_c[:, None, None] * \
                                 torch.einsum('ni,nj->nij',
                                              (substraction_save(kalman_sigmaPoints, var_p31),
                                               var_u2)), 0).data)
    var_u33[:, ~x_t_mask] = 0.0 # according to 6.22 (assuming C <=> P^t-1_t A^'_t) in Shumway et al. (i.e. after substition of A, c.f. 6.77 & 6.78)
    
    # u.4.1 (i.e. K) [Kalman gain]
    var_u32.data.copy_(torch.inverse(var_u32).data)
    var_u32.data.copy_((0.5 * (var_u32 + var_u32.transpose(1, 0))).data)
    var_u41.data.copy_(torch.mm(var_u33, var_u32).data)
    
    # var_u31 is treated as dummy to save intermediate calculations
    var_u31.data.copy_(substraction_save(x_t, var_u31).data)
    var_u31[~x_t_mask] = 0.0 # according to 6.23 (epsilon used in 6.20) in Shumway et al. (i.e. after substition of y and A, c.f. 6.77 & 6.78) [use estimated position as measurement when no measurement is available]

    # u.4.2
    mu_ukf.data.copy_(addition_save(var_p31,
                                    torch.mv(var_u41, var_u31)).data)

    # u.4.3 [P - K S K^T = P - C K^T = P - K C^T since K = C S^-1 and S = S^T => S^-1 = S^-T]
    var_ukf.data.copy_(substraction_save(var_p32,
                                         torch.mm(var_u41,
                                                  var_u33.transpose(1, 0))).data)
    var_ukf.data.copy_((0.5 * (var_ukf + var_ukf.transpose(1, 0))).data)
    return
