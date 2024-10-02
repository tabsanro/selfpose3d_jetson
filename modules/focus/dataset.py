from argparse import Namespace
import copy
import os
import pickle
from typing import List

import cv2
import numpy as np
from torch.utils.data import IterableDataset
import torchvision.transforms as transforms

from .get_frames import get_frames
from SelfPose3d.lib.utils.transforms import get_affine_transform, get_scale


class FOCUSDataset(IterableDataset):
    def __init__(self, cfg, sources, calib_path, pipelines, device='cpu'):
        self.device = device
        self.sources=sources
        self.pipelines = pipelines
        self.calib_path=calib_path
        self.num_sources = cfg.NUM_SOURCES
        self.transform = self.get_transform()
        self.image_size = np.array([960, 512])
        self.camera_data = self.load_camera_data(self.calib_path, self.pipelines)
        self.frames_generator = get_frames(cfg, self.sources, self.pipelines)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        frames = next(self.frames_generator, None)
        if frames is None:
            raise StopIteration
        # origin_frames = copy.deepcopy(frames)
        meta, transed_frames = self.get_meta_and_transed_frame(frames)
        if self.device == 'cuda':
            transed_frames = self.to_cuda(transed_frames)
        return frames, transed_frames, meta

    def to_cuda(self, transed_frames):
        transed_frames = [frame.cuda() for frame in transed_frames]
        return transed_frames
    
    def get_transform(self):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        return transform
    
    def get_meta_and_transed_frame(self, frames: List[np.ndarray]) -> List[dict]:
        meta = []
        transed_frames = []
        for i, frame in enumerate(frames):
            height, width, _ = frame.shape
            center = np.array([width / 2, height / 2])
            scale = get_scale((width, height), self.image_size)
            rotation = 0
            # meta
            meta.append({
                'center': center,
                'scale': scale,
                'rotation': rotation,
                'camera': self.camera_data[i]
            })
            # tensor
            trans = get_affine_transform(center, scale, rotation, self.image_size)
            input = cv2.warpAffine(frame, trans, (int(self.image_size[0]), int(self.image_size[1])), flags=cv2.INTER_LINEAR)
            input = self.transform(input)
            transed_frames.append(input)
        return meta, transed_frames
        

    def load_camera_data(self, calib_path: str, pipelines: dict) -> List[dict]:
        data_list = []
        if pipelines is not None:
            for pipeline in pipelines:
                camera_data = pipelines[pipeline]
                R, _ = cv2.Rodrigues(camera_data['rvec'])
                T = (-np.dot(R.T, camera_data['tvec']) * 1000)
                cam = {
                    'R': R,
                    'T': T,
                    'fx': camera_data['camera_matrix'][0, 0],
                    'fy': camera_data['camera_matrix'][1, 1],
                    'cx': camera_data['camera_matrix'][0, 2],
                    'cy': camera_data['camera_matrix'][1, 2],
                    'k': camera_data['dist_coeffs'][0][[0,1,4]].reshape(3, 1),
                    'p': camera_data['dist_coeffs'][0][[2,3]].reshape(2, 1)
                }
                data_list.append(cam)
            return data_list

        for i in range(self.num_sources):
            try:
                with open(os.path.join(calib_path, f'camera{i+1}.pkl'), 'rb') as f:
                    camera_data = pickle.load(f)
            except (IOError, FileNotFoundError):
                raise FileNotFoundError(f'Camera calibration file not found: {os.path.join(calib_path, f"camera{i+1}.pkl")}')
            # M = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
            R, _ = cv2.Rodrigues(camera_data['rvec'])
            # R = R.dot(M)
            T = (-np.dot(R.T, camera_data['tvec']) * 1000)

            # Create camera dictionary
            cam = {
                'R': R,
                'T': T,  # Multiply by 1000 to match the unit of camera_data['tvec'] (assumed to be in meters)
                'fx': camera_data['camera_matrix'][0, 0],
                'fy': camera_data['camera_matrix'][1, 1],
                'cx': camera_data['camera_matrix'][0, 2],
                'cy': camera_data['camera_matrix'][1, 2],
                'k': camera_data['dist_coeffs'][0][[0,1,4]].reshape(3, 1),  # Distortion coefficients k1, k2, k3
                'p': camera_data['dist_coeffs'][0][[2,3]].reshape(2, 1)  # Distortion coefficients p1, p2
            }
            data_list.append(cam)
        return data_list

    def update_camera_data(self):
        """
        Update camera calibration data.

        TODO: Implement this function.

        """
        pass
