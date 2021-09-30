#!/usr/bin/env python3

import os

import configuration as cfg

if __name__ == "__main__":
    run_pose = True
    
    if os.path.isdir(cfg.folder_save):
        invalid_input = True
        while (invalid_input):
            print('The following target folder already exists:')
            print(cfg.folder_save)
            print('Do you want to overwrite the existing folder? [y/n]')
            input_user = input()
            if ((input_user == 'Y') or (input_user == 'y')):
                run_pose = True
                invalid_input = False
            elif ((input_user == 'N') or (input_user == 'n')):
                run_pose = False
                invalid_input = False
    else:
        # create target save folder
        cmd = 'mkdir "{:s}"'.format(cfg.folder_save)
        os.system(cmd)
        # create target save folder
        cmd = 'mkdir "{:s}"'.format(cfg.folder_save + '/evaluation')
        os.system(cmd)

    if (run_pose):
        # copy relevant import files
        cmd = 'cp run.py "{:s}/run.py"'.format(cfg.folder_save)
        os.system(cmd)
        cmd = 'cp routines_math.py "{:s}/routines_math.py"'.format(cfg.folder_save)
        os.system(cmd)
        cmd = 'cp anatomy.py "{:s}/anatomy.py"'.format(cfg.folder_save)
        os.system(cmd)
        cmd = 'cp configuration.py "{:s}/configuration.py"'.format(cfg.folder_save)
        os.system(cmd)
        cmd = 'cp helper.py "{:s}/helper.py"'.format(cfg.folder_save)
        os.system(cmd)
        cmd = 'cp model.py "{:s}/model.py"'.format(cfg.folder_save)
        os.system(cmd)
        cmd = 'cp optimization.py "{:s}/optimization.py"'.format(cfg.folder_save)
        os.system(cmd)
        cmd = 'cp interp_3d.py "{:s}/interp_3d.py"'.format(cfg.folder_save)
        os.system(cmd)
        # calibration
        cmd = 'cp calibration.py "{:s}/calibration.py"'.format(cfg.folder_save)
        os.system(cmd)
        # initialization
        cmd = 'cp initialization.py "{:s}/initialization.py"'.format(cfg.folder_save)
        os.system(cmd)
        # EM
        cmd = 'cp kalman.py "{:s}/kalman.py"'.format(cfg.folder_save)
        os.system(cmd)
        cmd = 'cp em.py "{:s}/em.py"'.format(cfg.folder_save)
        os.system(cmd)
        cmd = 'cp em_run.py "{:s}/em_run.py"'.format(cfg.folder_save)
        os.system(cmd)
        
        # change directory to save folder
        os.chdir(cfg.folder_save)
        
        # calibrate
        cmd = './calibration.py'
        os.system(cmd)
        # initialize
        cmd = './initialization.py'
        os.system(cmd)
        # run pose reconstruction
        if (cfg.mode == 4):
            # run probabilistic model
            cmd = './em_run.py'
            os.system(cmd)
        else:
            print('ERROR: Pease choose mode 4')
