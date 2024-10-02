import argparse
from datetime import datetime
import os
import time

import cv2
from tqdm import tqdm

import _init_paths
from config import config, update_config
from modules.focus.get_frames import get_frames

'''
Record multi-synchronized videos for pose inference
Videos will be saved in /data/focus-dataset/data/YYMMDD_HHMM/hdVideos
The format of the video file names is "hd_00_01.mp4"
'''

def get_parser():
    parser = argparse.ArgumentParser(description="Record AISL dataset")
    parser.add_argument("--num-sources", "-N", type=int, default=4, help="Number of cameras to record from.")
    parser.add_argument("--is-kafka", action="store_true", help="Use kafka instead of cameras.")
    parser.add_argument("--duration", type=int, default=15*60, help="Duration of recording in seconds.")
    parser.add_argument("--imshow", action="store_true", help="Show the video feed.")
    parser.add_argument("--fourcc", type=str, default="mp4v", help="FourCC codec to use for video recording.")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for video recording.")
    parser.add_argument("--width", type=int, default=1920, help="Width of the video frame.")
    parser.add_argument("--height", type=int, default=1080, help="Height of the video frame.")
    parser.add_argument("--device-list", type=list, default=None, help="Device number of the camera. Starting from 1.")
    parser.add_argument("--server", type=str, default="localhost:9092", help="Kafka server address.")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    # Update the config file
    update_config('configs/focus.yaml')
    ''' Update the config file, and following lines are the original content of the config file
        NUM_SOURCES: 4
        KAFKA:
            CAMERA:
                ENABLED: True
                GROUP_ID: camera
                TOPICS: ~
            SERVER: localhost:9092
    '''
    config.NUM_SOURCES = args.num_sources
    config.KAFKA.CAMERA.ENABLED = args.is_kafka
    if args.device_list:
        config.KAFKA.CAMERA.TOPICS = [f'camera{i}' for i in args.device_list]
    else:
        config.KAFKA.CAMERA.TOPICS = [f'camera{i+1}' for i in range(args.num_sources)]
    config.KAFKA.SERVER = args.server
    
    # Video recording
    today = datetime.today().strftime('%y%m%d_%H%M')
    is_recording = True
    output_dir = os.path.join('data', 'focus-dataset', 'data', today, 'hdVideos')

    os.makedirs(output_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*args.fourcc)
    outs = [
        cv2.VideoWriter(
            os.path.join(output_dir, f'hd_00_{i+1:02d}.mp4'),
            fourcc,
            args.fps,
            (args.width, args.height),
        ) for i in range(args.num_sources)
    ]
    
    # Kafka consumer or local camera
    frame_generator = get_frames(config, None)
    
    start_time = cv2.getTickCount()
    while is_recording and (cv2.getTickCount() - start_time) / cv2.getTickFrequency() < args.duration:
        frames = next(frame_generator, None)
        if frames is None:
            break
        for cam_idx, (out, frame) in enumerate(zip(outs, frames)):
            out.write(frame)
            if args.imshow:
                cv2.imshow(f'Camera {cam_idx}', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    for out in outs:
        out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
