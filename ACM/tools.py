import os
import numpy as np
import torch
from scipy.io import savemat
import shutil

#from . import em
from . import helper
#from . import kalman
from . import model


def propagate_latent_to_pose(config,save_dict,x_ini):

    mu_ini = x_ini
    if ('mu_uks' in save_dict):
        mu_uks = save_dict['mu_uks'][1:]
        var_uks = save_dict['var_uks'][1:]
        print(save_dict['message'])
    else:
        mu_uks = save_dict['mu'][1:]
        nPara = np.size(mu_uks, 1)
        var_dummy = np.identity(nPara, dtype=np.float64) * 2**-52
        var_uks = np.tile(var_dummy.ravel(), config['nT']).reshape(config['nT'], nPara, nPara)

    nT = np.size(mu_uks, 0)

    file_origin_coord = config['file_origin_coord']
    file_calibration = config['file_calibration']
    file_model = config['file_model']
    file_labelsDLC = config['file_labelsDLC']

    args = helper.get_arguments(file_origin_coord, file_calibration, file_model, file_labelsDLC,
                                config['scale_factor'], config['pcutoff'])
    if ((config['mode'] == 1) or (config['mode'] == 2)):
        args['use_custom_clip'] = False
    elif ((config['mode'] == 3) or (config['mode'] == 4)):
        args['use_custom_clip'] = True
    args['plot'] = True
    del(args['model']['surface_vertices'])
    joint_order = args['model']['joint_order']
    joint_marker_order = args['model']['joint_marker_order']
    skeleton_edges = args['model']['skeleton_edges'].cpu().numpy()
    nCameras = args['numbers']['nCameras']
    nMarkers = args['numbers']['nMarkers']
    nBones = args['numbers']['nBones']

    free_para_bones = args['free_para_bones'].cpu().numpy()
    free_para_markers = args['free_para_markers'].cpu().numpy()
    free_para_pose = args['free_para_pose'].cpu().numpy()
    free_para_bones = np.zeros_like(free_para_bones, dtype=bool)
    free_para_markers = np.zeros_like(free_para_markers, dtype=bool)
    nFree_bones = int(0)
    nFree_markers = int(0)
    free_para = np.concatenate([free_para_bones,
                                free_para_markers,
                                free_para_pose], 0)
    args['x_torch'] = torch.from_numpy(mu_ini).type(model.float_type)
    args['x_free_torch'] = torch.from_numpy(mu_ini[free_para]).type(model.float_type)
    args['free_para_bones'] = torch.from_numpy(free_para_bones)
    args['free_para_markers'] = torch.from_numpy(free_para_markers)
    args['nFree_bones'] = nFree_bones
    args['nFree_markers'] = nFree_markers
    args['x_torch'] = torch.from_numpy(mu_ini).type(model.float_type)
    args['x_free_torch'] = torch.from_numpy(mu_ini[free_para]).type(model.float_type)
    #
    z_all = torch.from_numpy(mu_uks)

    marker_proj, marker_pos, skel3d_all = model.fcn_emission_free(z_all, args)

    pose = {
        'marker_positions_2d': marker_proj.detach().cpu().numpy().reshape(nT, nCameras, nMarkers, 2),
        'marker_positions_3d': marker_pos.detach().cpu().numpy(),
        'joint_positions_3d': skel3d_all.cpu().numpy(),
        }

    return pose

def copy_config(config,input_path):
    configdocdir = config["folder_save"]+'/configuration/'
    os.makedirs(configdocdir,exist_ok=True)

    shutil.copy(input_path+'/configuration.py',configdocdir)
    shutil.copyfile(config['file_origin_coord'],configdocdir+"/file_origin_coord.npy")
    shutil.copyfile(config['file_calibration'],configdocdir+"/file_calibration.npy")
    shutil.copyfile(config['file_model'],configdocdir+"/file_model.npy")
    shutil.copyfile(config['file_labelsDLC'],configdocdir+"/file_labelsDLC.npy")
    shutil.copyfile(config['file_labelsManual'],configdocdir+"/file_labelsManual.npz")
