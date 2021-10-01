#!/usr/bin/env python3
import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="ACM (Anatomically-constrained model) - a framework for videography based pose tracking of rodents")
    parser.add_argument('INPUT_PATH', type=str, help="Directory with job configuration")
    args = parser.parse_args()

    input_path = os.path.expanduser(args.INPUT_PATH)
    print(f'Processing {input_path} ...')
    sys.path.insert(0,input_path)

    # TODO change config system, e.g. pass around a dictionary instead of importing the config everywhere, requiring the sys.path.insert
    import configuration as cfg

    from . import calibration
    from . import initialization
    from . import em_run

    print(f'Saving to {cfg.folder_save}')

    run_pose = True
    
    if os.path.isdir(cfg.folder_save):
        invalid_input = True
        while (invalid_input):
            print('Target folder already exists. Do you want to overwrite the existing folder? [y/n]')
            input_user = input()
            if ((input_user == 'Y') or (input_user == 'y')):
                run_pose = True
                invalid_input = False
            elif ((input_user == 'N') or (input_user == 'n')):
                run_pose = False
                invalid_input = False
    else:
        # create target save folder
        os.makedirs(cfg.folder_save)
        #os.makedirs(cfg.folder_save + '/evaluation')

    if (run_pose):
        # calibrate
        calibration.main()
        # initialize
        initialization.main()
        # run pose reconstruction
        if (cfg.mode == 4):
            # run probabilistic model
            em_run.main()
        else:
            print('ERROR: Please choose mode 4')


if __name__ == "__main__":
    main()
