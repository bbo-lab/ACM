Detection of LED head stage based on MTT files, used by the BBO lab at caesar research center

# Installation

## Windows

1. [Install Anaconda](https://docs.anaconda.com/anaconda/install/windows/)
2. Clone git@github.com:bbo-lab/multitrackpy.git 
3. Open Anaconda prompt via Start Menu
4. Using `cd` and `dir`, navigate to the multitrackpy folder INSIDE the repository (which may also be named multitrackpy)
5. Create conda environment using `conda env create -f environment.yml`
6. Switch to multitrackpy environment: `conda activate multitrackpy`
7. Add multitrackpy module to conda environment: `conda develop [path to your repository, including repository folder]`

You can now run the program with `python -m multitrackpy -h`:
```
usage: __main__.py [-h] --mtt_file MTT_FILE --video_dir VIDEO_DIR
                   [--linedist_thres LINEDIST_THRES] [--corr_thres CORR_THRES]
                   [--led_thres LED_THRES] [--n_cpu N_CPU]
                   START_IDX END_IDX
__main__.py: error: the following arguments are required: START_IDX, END_IDX, --mtt_file, --video_dir
```
