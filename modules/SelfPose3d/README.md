## Introduction


This folder contains a simplified version of the code for the AISL Digital Twin team to use the selfpose3D model for inference purposes. All executable files and data, except for the `/model` folder, are prepared in the `/aisl` folder. If you need help with data preparation and running examples, please refer to the following YouTube video or follow the subsequent instructions.


## Data preparation
```
python SelfPose3d/tools/download_from_drive.py
```
or
```
python quick_setup.py
```


This will download three folders into your project folder: hdVideos, calibration and models.

The directory tree should look like this:
```
${POSE_ROOT}
|-- SelfPose3d
|   |-- aisl
|   |   |-- data_0705
|   |   |   |-- hdVideos
|   |   |   |   |-- hd_00_01.mp4
                ...
|   |   |   |   |-- hd_00_04.mp4
|   |   |   |-- calibration
|   |   |   |   |-- camera1.pkl
                ...
|   |   |   |   |-- camera4.pkl
|   |-- models
|   |   |-- cam5_rootnet_epoch2.pth.tar
|   |   |-- cam5_posenet.pth.tar
|   |   |-- backbone_epoch20.pth.tar

```

## how to run
