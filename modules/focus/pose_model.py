import os

import torch
import torch.nn as nn

# from SelfPose3d import load_model
from SelfPose3d.lib.models.pose_resnet import get_pose_net
from SelfPose3d.lib.models.engine_model import EngineModel
from SelfPose3d.lib.models.multi_person_posenet_ssv import MultiPersonPoseNetSSV as PoseNet
from SelfPose3d.lib.core.config import config as cfg, update_config as update_cfg

class PoseModel(nn.Module):
    @torch.no_grad()
    def __init__(
        self,
        gpus,
        root,
        config,
    ):
        # config: focus, cfg: selfpose3d
        super().__init__()
        sp3d_config = os.path.join(root, '..', config.CONFIG.POSENET)
        update_cfg(sp3d_config)
        
        torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED                # set to False if you want to disable cuDNN
        torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK            # set to True if input sizes are the same
        torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC    # set to True if you want reproducible results
        
        self.gpus = gpus
        self.sp3d_ckpt = os.path.join(root, '..', config.MODEL.POSENET.CKPT)
        
        self.engine_path = os.path.join(root, '..', 'modules', 'SelfPose3d', 'models', 'backbone.engine')
        self.tensorrt = config.MODEL.POSENET.TENSORRT
        
        self.model = self.load_model()
        self.model.eval()
    
    def forward(self, raw_images, views1, meta1):
        # inference
        results = self.model(raw_images=raw_images, views1=views1, meta1=meta1, distance=None)
        
        # unpack
        pred_3d, pred_2d, all_heatmaps, grid_centers, face_images = results
        
        # delete None values
        # TODO pred_2d
        pred_3d = pred_3d[pred_3d[...,3]==0]
        grid_centers = grid_centers[grid_centers[...,3]==0]
        
        # check None values and delete batch dimension
        if pred_3d.shape[0] == 0:
            pred_3d = None
        else:
            pred_3d = pred_3d.squeeze(0).view(-1, 15, 5)  # (batch, n, 15, 5) -> (n, 15, 5)
            pred_2d = pred_2d.squeeze(0).view(-1, 15, 2)  # (batch, n, c, 15, 2) -> (n, c, 15, 2)
        if grid_centers.shape[0] == 0:
            grid_centers = None
        else:
            grid_centers = grid_centers.squeeze(0).view(-1, 5)  # (batch, n, 5) -> (n, 5)
        
        return (pred_3d, pred_2d, all_heatmaps, grid_centers, face_images)
    
    def load_model(self):
        backbone = self.load_backbone(is_train=True)
        model = PoseNet(backbone, cfg)
        model = nn.DataParallel(model, device_ids=self.gpus).cuda()
        model.module.load_state_dict(torch.load(self.sp3d_ckpt), strict=False)
        return model
    
    def load_backbone(self, is_train=True):
        if self.tensorrt:
            return EngineModel(self.engine_path, copy=True)
        return get_pose_net(cfg, is_train=is_train)
