# Dataset

This page shows instructions on how to download the process the data used in this paper. After accquiring the data, we expect the directory structure of the data to be:
```
data/
├── zju/
    ├── SMPL_NEUTRAL.pkl
    ├── CoreView_313/
    ├── ...
    └── CoreView_386/
```

## ZJU Mocap Dataset

Step 1. As we are not allowed to redistributed this dataset, please request the data from the ZJU authors following the instructions [here](https://github.com/zju3dv/neuralbody/blob/master/INSTALL.md#zju-mocap-dataset). After downloading the data, please put it under `./data` folder as instructed above.

Step 2. Download SMPL neural body model from [SMPLify website](https://smplify.is.tue.mpg.de/index.html). Rename the file `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` into `SMPL_NEUTRAL.pkl` and put it under `./data/` folder as instructed above.

Step 3. Process the data to extract skeleton information. This script will create a file `CoreView_XXX/pose_data.pt` for each subject, which stores the skeleton SE(4) transformations etc used for AV3D and baseline methods:
```
python tools/process_zju/main.py
```

To convert `mdm.npy` to later animate the reconstructed avatar, remember to switch the dataloader correctly under `av3d/configs/dataset/zju.yaml`

```
python tools/process_zju/main_mdm.py
```
