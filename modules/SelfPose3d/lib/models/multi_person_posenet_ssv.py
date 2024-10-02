# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from copy import deepcopy
from models import pose_resnet
from models.cuboid_proposal_net_soft import CuboidProposalNetSoft
from models.pose_regression_net import PoseRegressionNet
import torch.nn.functional as F
import numpy as np
import cv2

import utils.cameras as cameras
from utils.transforms import get_affine_transform, get_scale


class MultiPersonPoseNetSSV(nn.Module):
    def __init__(self, backbone, cfg):
        super(MultiPersonPoseNetSSV, self).__init__()
        self.backbone = backbone
        self.pose_net = PoseRegressionNet(cfg)
        self.root_net = CuboidProposalNetSoft(cfg)
        self.num_joints = cfg.NETWORK.NUM_JOINTS
        self.num_cand = cfg.MULTI_PERSON.MAX_PEOPLE_NUM
        self.image_size = (256, 256)
        self.crop_bound = 'face'

    def _2d_projection(self, pose_3d, meta):
        batch_size = pose_3d.shape[0]
        num_views = len(meta)
        pred_2d = torch.zeros(batch_size, num_views, self.num_joints, 2, device=pose_3d.device)
        for i in range(batch_size):
            for c in range(num_views):
                cam = {}
                center = meta[c]['center'][i]
                width, height = center * 2
                for k, v in meta[c]['camera'].items():
                        cam[k] = v[i]
                pose_2d = cameras.project_pose(pose_3d[i], cam)
                # pose_2d[:,0] = torch.clamp(pose_2d[:,0],0,height-1)
                # pose_2d[:,1] = torch.clamp(pose_2d[:,1],0,width-1)
                pred_2d[i, c] = pose_2d[:, :2]

        return pred_2d

    def _crop_face_images(self, raw_images, pred_2d):
        batch_size, num_views, _, _ = pred_2d.shape
        device = pred_2d.device
        
        # 결과 저장을 위한 텐서 초기화
        face_images = [[] for _ in range(batch_size)]
        
        for b in range(batch_size):
            for v in range(num_views):
                # 현재 배치와 뷰에 대한 2D 예측 좌표
                cur_pred_2d = pred_2d[b, v]
                
                # 얼굴 크롭
                max_x = int((cur_pred_2d[1, 0]).item()) + 112
                min_x = int((cur_pred_2d[1, 0]).item()) - 111
                max_y = int((cur_pred_2d[1, 1]).item()) + 112
                min_y = int((cur_pred_2d[1, 1]).item()) - 111
                
                # 원본 이미지에서 관심 영역 추출
                # img = raw_images[v][b]
                img = raw_images[b][v]  # (b, cam, h, w, ch)
                
                # 이미지 경계 확인
                h, w, _ = img.shape
                min_x, min_y = max(0, min_x), max(0, min_y)
                max_x, max_y = min(w-1, max_x), min(h-1, max_y)
                
                cropped = img[min_y:max_y+1, min_x:max_x+1, :]
                face_images[b].append(cropped)
                
        return face_images
    
    def _cal_distance(self, root, distance):
        if distance is None or distance == 0:
            return True
        d = torch.norm(root).item()
        return torch.norm(root).item() < distance

    def forward(
        self,
        raw_images=None,
        views1=None,
        meta1=None,
        distance=None,
    ):
        all_heatmaps = []
        for view in views1:
            heatmaps = self.backbone(view)
            all_heatmaps.append(heatmaps)

        device = all_heatmaps[0].device
        batch_size = all_heatmaps[0].shape[0]

        _, _, _, grid_centers = self.root_net(all_heatmaps, meta1)

        pred = torch.zeros(batch_size, self.num_cand, self.num_joints, 5, device=device)
        pred[:, :, :, 3:] = grid_centers[:, :, 3:].reshape(batch_size, -1, 1, 2)

        pred_2d = torch.zeros(batch_size, self.num_cand, len(meta1), self.num_joints, 2, device=device)
        crop_face_images = []

        if not isinstance(raw_images, np.ndarray): # (b, cam, h, w, ch)
            raw_images = np.array(raw_images.detach().cpu())
            # raw_images = [img.detach().cpu().numpy() for img in raw_images]

        for n in range(self.num_cand):
            index = pred[:, n, 0, 3] >= 0
            if torch.sum(index) > 0:
                # grid_center shape : (b, n, 5), (1, 10, 5)
                if self._cal_distance(grid_centers[:, n, :2], distance) == False:
                    grid_centers[:, n, 3] = 1
                    pred[:, n, :, 3] = 1
                    continue
                single_pose = self.pose_net(all_heatmaps, meta1, grid_centers[:, n])
                if min(single_pose[:,8,2], single_pose[:,14,2]) < 0 or min(single_pose[:,8,2], single_pose[:,14,2]) > 120:
                    grid_centers[:, n, 3] = -1
                    pred[:, n, :, 3] = -1
                    continue
                pred_2d[:, n, :, :, :] = self._2d_projection(single_pose, meta1)
                crop_face_images.append(self._crop_face_images(raw_images, pred_2d[:, n, :, :]))

                pred[:, n, :, 0:3] = single_pose.detach()
                del single_pose

        return pred, pred_2d, all_heatmaps, grid_centers, crop_face_images


def  get_multi_person_pose_net(cfg, is_train=True, tensorrt=False, engine_path=None):
    if tensorrt:
        from .engine_model import EngineModel
        backbone = EngineModel(engine_path, copy=True)
    else:
        backbone = eval(cfg.BACKBONE_MODEL + ".get_pose_net")(cfg, is_train=is_train)
    model = MultiPersonPoseNetSSV(backbone, cfg)
    return model