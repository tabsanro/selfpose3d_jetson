import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from easydict import EasyDict as edict
import numpy as np
import yaml

# from SelfPose3d.lib.core.config import config as cfg

config = edict()
config.NUM_SOURCES = 4
config.WEBCAM = False
config.SOURCE_FOLDER_NAME = None
config.MIVOLO = True
config.MIVOLO_TRACK = True
config.MAIN = True

# model
config.MODEL = edict()
config.MODEL.POSENET = edict()
config.MODEL.POSENET.CKPT = "modules/SelfPose3d/models/POC_posenet.pth.tar"
config.MODEL.POSENET.TENSORRT = False
config.MODEL.POSENET.INPUT_SHAPE = None

config.MODEL.MIVOLO = edict()
config.MODEL.MIVOLO.CKPT = "modules/mivolo/models/model_imdb_cross_person_4.22_99.46.pth.tar"
config.MODEL.MIVOLO.TENSORRT = False
config.MODEL.MIVOLO.INPUT_SHAPE = None

config.MODEL.YOLO = edict()
config.MODEL.YOLO.CKPT = "modules/mivolo/models/yolov8x_person_face.pt"
config.MODEL.YOLO.TENSORRT = False
config.MODEL.YOLO.INPUT_SHAPE = None

# model configs
config.CONFIG = edict()
config.CONFIG.POSENET = "modules/SelfPose3d/config/cam4_posenet.yaml"

# Camera
config.CAMERA = edict()
config.CAMERA.SOURCES = "modules/SelfPose3d/data0705/hdVideo"
config.CAMERA.START_FRAME = 0
config.CAMERA.END_FRAME = None
config.CAMERA.MAX_CAMERAS = 100

# lod
config.DISTANCE = 1500

# kafka
config.KAFKA = edict()
config.KAFKA.SERVER = "localhost:9092"

# camera
config.KAFKA.CAMERA = edict()
config.KAFKA.CAMERA.ENABLED = False
config.KAFKA.CAMERA.GROUP_ID = "camera"
config.KAFKA.CAMERA.TOPICS = None

# analytics
config.KAFKA.ANALYTICS = edict()
config.KAFKA.ANALYTICS.ENABLED = False
config.KAFKA.ANALYTICS.GROUP_ID = edict()
config.KAFKA.ANALYTICS.GROUP_ID.MAIN = "main"
config.KAFKA.ANALYTICS.GROUP_ID.SUB = "sub"

config.KAFKA.ANALYTICS.TOPIC = edict()
config.KAFKA.ANALYTICS.TOPIC.REQUEST = "request"
config.KAFKA.ANALYTICS.TOPIC.RESPONSE = "response"

# visualization
config.KAFKA.VISUALIZATION = edict()
config.KAFKA.VISUALIZATION.ENABLED = False
config.KAFKA.VISUALIZATION.GROUP_ID = "aisl_sim"
config.KAFKA.VISUALIZATION.TOPIC = "person_info"

# control
config.KAFKA.CONTROL = edict()
config.KAFKA.CONTROL.ENABLED = False
config.KAFKA.CONTROL.GROUP_ID = "control"
config.KAFKA.CONTROL.TOPIC = "control"


def _update_dict(k, v):
    if k == 'DATASET':
        if 'MEAN' in v and v['MEAN']:
            v['MEAN'] = np.array(
                [eval(x) if isinstance(x, str) else x for x in v['MEAN']])
        if 'STD' in v and v['STD']:
            v['STD'] = np.array(
                [eval(x) if isinstance(x, str) else x for x in v['STD']])
    if k == 'NETWORK':
        if 'HEATMAP_SIZE' in v:
            if isinstance(v['HEATMAP_SIZE'], int):
                v['HEATMAP_SIZE'] = np.array(
                    [v['HEATMAP_SIZE'], v['HEATMAP_SIZE']])
            else:
                v['HEATMAP_SIZE'] = np.array(v['HEATMAP_SIZE'])
        if 'IMAGE_SIZE' in v:
            if isinstance(v['IMAGE_SIZE'], int):
                v['IMAGE_SIZE'] = np.array([v['IMAGE_SIZE'], v['IMAGE_SIZE']])
            else:
                v['IMAGE_SIZE'] = np.array(v['IMAGE_SIZE'])
    # add new keys
    if k not in config or not isinstance(config[k], dict):
        config[k] = edict()
    for vk, vv in v.items():
        config[k][vk] = vv
        # if vk in config[k]:
        #     config[k][vk] = vv
        # else:
        #     raise ValueError("{}.{} not exist in config.py".format(k, vk))

def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(k, v)
                else:
                    if k == 'SCALES':
                        config[k][0] = (tuple(v))
                    else:
                        config[k] = v
            else:
                # raise ValueError("{} not exist in config.py".format(k))
                if isinstance(v, dict):
                    config[k] = edict(v)
                else:
                    config[k] = v
