#!/usr/bin/env python3
import argparse
import os
import sys
from pprint import pprint
import numpy as np
from scipy.io import savemat

def main():
    # Parse inputs
    parser = argparse.ArgumentParser(description="ACM (Anatomically-constrained model) - a framework for videography based pose tracking of rodents")
    parser.add_argument('INPUT_PATH', type=str, help="Directory with job configuration")
    parser.add_argument('--viewer', required=False, help="Load viewer instead of tracking pose", action="store_true")
    parser.add_argument('--export', required=False, help="Exports result in alternative format", action="store_true")
    parser.add_argument('--makepose', required=False, help="Creates pose file save_dict", action="store_true")
    parser.add_argument('--calibration', required=False, help="Perform calibration only", action="store_true")
    parser.add_argument('--initialization', required=False, help="Perform initialization only (Requires calibration)", action="store_true")
    parser.add_argument('--poseinference', required=False, help="Perform poseinference only (Requires calibration and initialization)", action="store_true")
    args = parser.parse_args()
    input_path = os.path.expanduser(args.INPUT_PATH)

    # Load config
    # TODO change config system, e.g. pass around a dictionary instead of importing the config everywhere, requiring the sys.path.insert

    if args.viewer:
        sys.path.insert(0,input_path)
        print(f'Loading {input_path} ...')
        viewer()
    elif args.export:
        from .export import export
        export(input_path) 
    elif args.makepose:
        config_path = input_path+'/../..'
        sys.path.insert(0,config_path)
        from ACM import tools 
        print(f'Loading {config_path} ...')
        config = get_config_dict()
        save_dict = np.load(input_path+'/save_dict.npy',allow_pickle=True).item()
        x_ini = np.load(input_path+'/x_ini.npy',allow_pickle=True)
        pose = tools.propagate_latent_to_pose(config,save_dict,x_ini)

        posepath = input_path+'/pose'
        print(f'Saving pose to {posepath}')
        np.save(posepath+'.npy', pose)
        savemat(posepath+'.mat', pose)
    else:
        sys.path.insert(0,input_path)
        from ACM.export import export
        from ACM.tools import copy_config
        print(f'Loading {input_path} ...')
        check(args)
        config = get_config_dict()
        copy_config(config,input_path)
        track(args)
        export(config['folder_save']) 

def check(args):
    import configuration as cfg
    full_pipline = args.calibration==False and args.initialization==False and args.poseinference==False
    
    if args.calibration==True or full_pipline:
        check_directory(cfg.folder_calib,'Calibration') or sys.exit(1)
    if args.initialization==True or full_pipline:
        check_directory(cfg.folder_init,'Initialization') or sys.exit(1)
    if args.poseinference==True or full_pipline:
        check_directory(cfg.folder_save,'Result') or sys.exit(1)

def track(args):
    import configuration as cfg

    from ACM import calibration
    from ACM import initialization
    from ACM import fitting
    from ACM import em_run
    
    full_pipline = args.calibration==False and args.initialization==False and args.poseinference==False
    
    # calibrate    
    if args.calibration==True or full_pipline:
        calibration.main()

    # initialize
    if args.initialization==True or full_pipline:
        initialization.main()
        
    # run pose reconstruction
    if args.poseinference==True or full_pipline:
        if ((cfg.mode == 1) or (cfg.mode == 2)):
            # run deterministic models
            fitting.main()
        elif ((cfg.mode == 3) or (cfg.mode == 4)):
            # run probabilistic models
            em_run.main()

def viewer():
    config = get_config_dict()

    from ACM.gui import viewer
    viewer.start(config)

def get_config_dict():
    import configuration as cfg

    config = vars(cfg)
    for k in list(config.keys()):
        if k.startswith('__'):
            del config[k]

    return config

def check_directory(path,dirtype):
    if os.path.isdir(path) :
        if len(os.listdir(path)) > 0:
            invalid_input = True
            while (invalid_input):
                print(f'{dirtype} folder {path} already exists. Do you want to overwrite the existing folder? [y/n]')
                input_user = input()
                if ((input_user == 'Y') or (input_user == 'y')):
                    run_pose = True
                    invalid_input = False
                elif ((input_user == 'N') or (input_user == 'n')):
                    return False
    else:
        # create target save folder
        os.makedirs(path)
    
    return True

if __name__ == "__main__":
    main()
