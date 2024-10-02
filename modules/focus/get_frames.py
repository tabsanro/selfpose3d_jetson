'''
This module provides a function to get frames from cameras or Kafka.

Usage:
    from focus.get_frames import get_frames
    frame_generator = get_frames(num_sources, is_kafka=False)
    while True:
        frames = next(frame_generator, None)
        if frames is None:
            break
        for i, frame in enumerate(frames):
            cv2.imshow(f'frame{i}', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
'''
from typing import Generator

import cv2
import numpy as np

from .camera import Camera

def get_frames(cfg, sources=None, pipelines=None) -> Generator[np.ndarray, None, None]:
    """
    Get frames from cameras or Kafka.

    Args:
        num_sources (int): The number of cameras.
        소스는 영상 쓸 때만 필요
        is_kafka (bool, optional): Whether to use Kafka.
        group_id (Optional[str], optional): Kafka group ID. Required when is_kafka is True.
        topics (Optional[List[str]], optional): Kafka topics list. Required when is_kafka is True.

    Yields:
        Generator: A generator of Numpy arrays of frames received from each source (camera or Kafka topic).
    """
    num_sources=cfg.NUM_SOURCES
    use_kafka=cfg.KAFKA.CAMERA.ENABLED
    group_id=cfg.KAFKA.CAMERA.GROUP_ID
    topics=cfg.KAFKA.CAMERA.TOPICS
    server=cfg.KAFKA.SERVER

    # if use_kafka:
    #     while True:
    #         yield None
    if pipelines is not None:
        while True:
            frames = []
            for i, pipeline in enumerate(pipelines):
                frame = pipeline.wait_for_frames()
                color_frame = frame.get_color_frame()
                if not color_frame:
                    raise IOError(f'Failed to get color frame from camera {i}')
                frame = np.asanyarray(color_frame.get_data())
                frames.append(frame)
            yield np.array(frames)
    else:
        try:
            camera = Camera(cfg, sources)
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
            yield np.array(frames)
