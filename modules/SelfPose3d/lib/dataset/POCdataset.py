import os.path as osp
import cv2
import pickle
import numpy as np
from torch.utils.data import IterableDataset
import torchvision.transforms as transforms
from utils.transforms import get_affine_transform, get_scale

class POCdatasetCV2(IterableDataset):
    def __init__(self, cfg, transform=None, start_idx=0, end_idx=None):
        _this_dir = osp.dirname(osp.abspath(__file__))
        self.data_dir = osp.abspath(osp.join(_this_dir, '../../data_0705'))
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.cam_list = [1,2,3,4]
        self.camera = self.get_cam()
        self.caps = self.get_caps()

        self.transform = self.get_transform()
        self.image_size = np.array(cfg.NETWORK.IMAGE_SIZE)
        
        if  self.end_idx is None:
            self.end_idx = int(self.caps[0].get(cv2.CAP_PROP_FRAME_COUNT))

    def get_transform(self):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        return transform

    def get_caps(self):
        caps = []
        video_path = self.data_dir + "/hdVideos/"
        for i in self.cam_list:
            cap = cv2.VideoCapture(video_path + f"hd_00_{i:02d}.mp4")
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_idx)
            caps.append(cap)
            
        return caps
        
    def get_cam(self):
        camera = []
        # for i in range(1,5):
        for i in self.cam_list:
            meta_file = self.data_dir + "/calibration/camera" + str(i) + ".pkl"
            with open(meta_file, "rb") as f:
                calib = pickle.load(f)
                
            M = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
            R, _ = cv2.Rodrigues(calib['rvec'])
            # R = R.dot(M)
            T = (
                -np.dot(R.T, calib['tvec']) * 1000
            )

            # m 딕셔너리 생성
            cam = {
                'R': R,
                'T': T,  
                'fx': calib['camera_matrix'][0, 0],
                'fy': calib['camera_matrix'][1, 1],
                'cx': calib['camera_matrix'][0, 2],
                'cy': calib['camera_matrix'][1, 2],
                'k': calib['dist_coeffs'][0][[0,1,4]].reshape(3, 1),  # 왜곡 계수 k1, k2, k3
                'p': calib['dist_coeffs'][0][[2,3]].reshape(2, 1)  # 왜곡 계수 p1, p2
            }
            camera.append(cam)
        return camera
    
    def __iter__(self):
        idx = 0     
        while True:
            if idx >= self.end_idx - self.start_idx:
                break
            idx += 1
            inputs = []
            raw_images = []
            for cap in self.caps:
                ret, data_numpy = cap.read()
                if not ret:
                    print(f"Failed to read image")
                    assert False
                
                data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)     

                height, width, _ = data_numpy.shape
                c = np.array([width / 2.0, height / 2.0])
                s = get_scale((width, height), self.image_size)
                r = 0

                trans = get_affine_transform(c, s, r, self.image_size)
                input = cv2.warpAffine(
                    data_numpy,
                    trans, (int(self.image_size[0]), int(self.image_size[1])),
                    flags=cv2.INTER_LINEAR)
                input = self.transform(input)

                inputs.append(input)
                raw_images.append(data_numpy)

            meta = []

            for i in range(len(self.cam_list)):
                # m 딕셔너리 생성
                m = {
                'center': c,
                'scale': s,
                'rotation': r,
                'camera': self.camera[i]
                }
                meta.append(m)
                
            yield raw_images, inputs, meta
        for cap in self.caps:
            cap.release()
