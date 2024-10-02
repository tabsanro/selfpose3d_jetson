'''
This module provides a Camera class for capturing frames from multiple cameras.

Usage:
    try:
        cameras = Camera(num_sources, sources=None, max_cameras=100)
    except ValueError as e:
        print(e)
        exit(1)
    while True:
        for i, cap in enumerate(cameras):
            ret, frame = cap.read()
            if ret:
                cv2.imshow(f"frame{i}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
'''

import cv2
import numpy as np
from tqdm import tqdm

class Camera:
    def __init__(self, cfg, sources):
        '''
        Initializes a Camera object.

        Args:
            num_sources (int): The number of cameras to capture frames from.
            sources (list, optional): List of camera indices to use. If not provided, the class will automatically find available cameras.
            max_cameras (int, optional): The maximum number of cameras to search for.

        Raises:
            ValueError: If the specified number of cameras cannot be found.

        '''
        self.caps = None
        self.num_sources = cfg.NUM_SOURCES
        self.cfg = cfg
        if sources:
            self.sources = sources
        else:
            self.sources = self.find_rgb_sources(cfg.CAMERA.MAX_CAMERAS)
        print(f"Found {len(self.sources)} cameras. {self.sources}")
        self.caps = self.get_capture()
        self.shape = self.__shape__()
    
    def __del__(self):
        '''
        Releases the camera captures when the object is deleted.
        '''
        if self.caps:
            for cap in self.caps:
                cap.release()
    
    def __shape__(self):
        '''
        Returns the shape of the camera captures.

        Returns:
            list: The shape of the camera captures. (width, height)

        '''
        return [(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) for cap in self.caps]
    
    def __len__(self):
        '''
        Returns the number of camera captures.

        Returns:
            int: The number of camera captures.

        '''
        return len(self.caps)
    
    def __getitem__(self):
        '''
        Returns the camera capture at the specified index.

        Args:
            idx (int): The index of the camera capture.

        Returns:
            cv2.VideoCapture: The camera capture object.

        '''
        return self.caps
    
    def __iter__(self):
        '''
        Returns an iterator over the camera captures.

        Returns:
            iterator: An iterator over the camera captures.

        '''
        return iter(self.caps)
    
    def find_rgb_sources(self, max_cameras):
        '''
        Finds available RGB camera sources.

        Args:
            max_cameras (int): The maximum number of cameras to search for.

        Returns:
            list: List of camera indices.

        Raises:
            ValueError: If the specified number of cameras cannot be found.

        '''
        sources = []
        for camera_idx in (camera_pbar := tqdm(range(max_cameras), desc="Finding cameras", unit="cameras")):
            if len(sources) == self.num_sources:
                camera_pbar.update(max_cameras-camera_idx)
                camera_pbar.close()
                break
            cap = cv2.VideoCapture(camera_idx)
            if cap.isOpened():
                for frame_idx in (frame_pbar := tqdm(range(10), desc=f"Checking camera {camera_idx}", unit="frames", leave=False)):
                    ret, frame = cap.read()
                    if not ret:
                        frame_pbar.update(10-frame_idx)
                        frame_pbar.close()
                        break
                cap.release()
                if ret:
                    if self.is_depth(frame):
                        continue
                    sources.append(camera_idx)
        if len(sources) != self.num_sources:
            raise ValueError(f"Failed to find {self.num_sources} cameras. Found {len(sources)} cameras.")
        return sources
    
    def get_capture(self):
        '''
        Returns a list of camera capture objects.

        Returns:
            list: List of camera capture objects.

        '''
        caps = []
        for source in self.sources:
            cap = cv2.VideoCapture(source)
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.cfg.CAMERA.START_FRAME)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            caps.append(cap)
        return caps
    
    def is_depth(self, frame):
        '''
        Checks if the frame is a depth frame and not an RGB frame.

        Args:
            frame (numpy.ndarray): The frame to check.

        Returns:
            bool: True if the frame is a depth frame, False otherwise.

        '''
        B, G, R = np.split(frame, indices_or_sections=3, axis=2)
        if np.array_equal(B, G) and np.array_equal(G, R):
            return True
        return False

if __name__ == "__main__":
    try:
        cameras = Camera(1)
    except ValueError as e:
        print(e)
        exit(1)
    print(cameras.shape)
    while True:
        for i, cap in enumerate(cameras):
            ret, frame = cap.read()
            if ret:
                cv2.imshow(f"frame{i}", frame)
        if cv2.waitKey(1000//15) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
