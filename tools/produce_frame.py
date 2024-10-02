import argparse

import cv2
import numpy as np

import _init_paths
from config import config, update_config
# import focus
from focus.camera import Camera
from focus.kafka_producer import KafkaProducer

def get_parser():
    parser = argparse.ArgumentParser(description="Camera Producer")
    parser.add_argument("--cfg", default="configs/focus.yaml", help="experiment configure file name", type=str)
    parser.add_argument("--device-list", type=list, default=None, help="Device number of the camera. Starting from 1.")
    args, rest = parser.parse_known_args()
    # we need following options
    # NUM_SOURCES
    # CAMERA.MAX_CAMERAS
    # KAFKA.SERVER
    update_config(args.cfg)
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    num_sources = config.NUM_SOURCES
    if args.device_list:
        topics = [f'camera{i}' for i in args.device_list]
    else:
        topics = [f'camera{i+1}' for i in range(num_sources)]
    producers = [KafkaProducer(config.KAFKA.SERVER, topic) for topic in topics]
    
    try:
        camera = Camera(config, None)
    except ValueError as e:
        print(e)
        exit(1)
    
    while True:
        frames = []
        for cap in camera:
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                raise IOError(f'Failed to read camera {cap}')
        for producer, frame in zip(producers, frames):
            producer.produce_message(frame)


if __name__ == '__main__':
    main()
