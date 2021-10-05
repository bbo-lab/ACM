#!/usr/bin/env python3
import argparse
import os
import sys
from pprint import pprint

def main():
    # Parse inputs
    parser = argparse.ArgumentParser(description="ACM (Anatomically-constrained model) - a framework for videography based pose tracking of rodents")
    parser.add_argument('INPUT_PATH', type=str, help="Directory with job configuration")
    parser.add_argument('--viewer', required=False, help="Load viewer instead of tracking pose", action="store_true")
    parser.add_argument('--calibration', required=False, help="Perform calibration only", action="store_true")
    parser.add_argument('--initialization', required=False, help="Perform initialization only (Requires calibration)", action="store_true")
    parser.add_argument('--poseinference', required=False, help="Perform poseinference only (Requires calibration and initialization)", action="store_true")
    args = parser.parse_args()
    input_path = os.path.expanduser(args.INPUT_PATH)

    # Load config
    # TODO change config system, e.g. pass around a dictionary instead of importing the config everywhere, requiring the sys.path.insert
    sys.path.insert(0,input_path)
    print(f'Loading {input_path} ...')

    if args.viewer:
        viewer()
    else:
        track(args)

def track(args):
    import configuration as cfg

    from . import calibration
    from . import initialization
    from . import em_run
    
    full_pipline = args.calibration==False and args.initialization==False  and args.poseinference==False

    # calibrate    
    if args.calibration==True or full_pipline:
        check_directory(cfg.folder_calib,'Calibration') or sys.exit(1)
        calibration.main()

    # initialize
    if args.initialization==True or full_pipline:
        check_directory(cfg.folder_init,'Initialization') or sys.exit(1)
        initialization.main()
        
    # run pose reconstruction
    if args.poseinference==True or full_pipline:
        check_directory(cfg.folder_save,'Result') or sys.exit(1)
        if (cfg.mode == 4):
            # run probabilistic model
            em_run.main()
        else:
            print('ERROR: Please choose mode 4')


def viewer():
    config = get_config_dict()

    from .gui import viewer
    viewer.start(config)


def get_config_dict():
    import configuration as cfg

    config = vars(cfg)
    for k in list(config.keys()):
        if k.startswith('__'):
            del config[k]

    return config


def check_directory(path,dirtype):
    if os.path.isdir(path):
        invalid_input = True
        while (invalid_input):
            print('{dirtype} folder already exists. Do you want to overwrite the existing folder? [y/n]')
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
