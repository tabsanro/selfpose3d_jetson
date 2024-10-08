# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from models.v2v_net import V2VNet
from models.project_layer import ProjectLayer
from core.proposal import nms
import numpy as np
from copy import deepcopy
import utils.cameras as cameras
import cv2


class ProposalLayerSoft(nn.Module):
    def __init__(self, cfg):
        super(ProposalLayerSoft, self).__init__()
        self.grid_size = torch.tensor(cfg.MULTI_PERSON.SPACE_SIZE)
        self.cube_size = torch.tensor(cfg.MULTI_PERSON.INITIAL_CUBE_SIZE)
        self.grid_center = torch.tensor(cfg.MULTI_PERSON.SPACE_CENTER)
        self.num_cand = cfg.MULTI_PERSON.MAX_PEOPLE_NUM
        self.root_id = cfg.DATASET.ROOTIDX
        self.num_joints = cfg.NETWORK.NUM_JOINTS
        self.threshold = cfg.MULTI_PERSON.THRESHOLD

    def filter_proposal(self, topk_index, gt_3d, num_person):
        batch_size = topk_index.shape[0]
        cand_num = topk_index.shape[1]
        cand2gt = torch.zeros(batch_size, cand_num)

        for i in range(batch_size):
            cand = topk_index[i].reshape(cand_num, 1, -1)
            gt = gt_3d[i, : num_person[i]].reshape(1, num_person[i], -1)

            dist = torch.sqrt(torch.sum((cand - gt) ** 2, dim=-1))
            min_dist, min_gt = torch.min(dist, dim=-1)

            cand2gt[i] = min_gt
            cand2gt[i][min_dist > 500.0] = -1.0

        return cand2gt

    def get_real_loc(self, index):
        device = index.device
        cube_size = self.cube_size.to(device=device, dtype=torch.float)
        grid_size = self.grid_size.to(device=device)
        grid_center = self.grid_center.to(device=device)
        loc = index.float() / (cube_size - 1) * grid_size + grid_center - grid_size / 2.0
        return loc

    def forward(self, root_cubes, meta, grids):
        batch_size = root_cubes.shape[0]

        topk_values, topk_unravel_index = nms(root_cubes.detach(), self.num_cand)
        topk_unravel_index = self.get_real_loc(topk_unravel_index)

        grid_centers = torch.zeros(batch_size, self.num_cand, 5, device=root_cubes.device)
        grid_centers[:, :, 0:3] = topk_unravel_index
        grid_centers[:, :, 4] = topk_values

        grid_centers[:, :, 3] = (
            topk_values > self.threshold
        ).float() - 1.0  # if ground-truths are not available.

        return grid_centers


class CuboidProposalNetSoft(nn.Module):
    def __init__(self, cfg):
        super(CuboidProposalNetSoft, self).__init__()
        self.grid_size = cfg.MULTI_PERSON.SPACE_SIZE
        self.cube_size = cfg.MULTI_PERSON.INITIAL_CUBE_SIZE
        self.grid_center = cfg.MULTI_PERSON.SPACE_CENTER
        self.root_id = cfg.DATASET.ROOTIDX
        self.rootnet_roothm = cfg.NETWORK.ROOTNET_ROOTHM
        self.rootnet_train_synth = cfg.NETWORK.ROOTNET_TRAIN_SYNTH
        self.max_num_people = cfg.MULTI_PERSON.MAX_PEOPLE_NUM

        self.project_layer = ProjectLayer(cfg)
        if self.rootnet_roothm:
            self.v2v_net = V2VNet(1, 1)
        else:
            self.v2v_net = V2VNet(cfg.NETWORK.NUM_JOINTS, 1)
        self.proposal_layer = ProposalLayerSoft(cfg)

    def get_grid_centres(self, all_heatmaps, meta, flip_xcoords):
        # with torch.no_grad():
        # self.v2v_net.eval()
        if self.rootnet_roothm:
            all_heatmaps_copy = [a[:, self.root_id, :, :][:, None].clone() for a in all_heatmaps]
        else:
            all_heatmaps_copy = all_heatmaps

        initial_cubes, grids = self.project_layer(
            all_heatmaps_copy,
            meta,
            self.grid_size,
            [self.grid_center],
            self.cube_size,
            flip_xcoords=flip_xcoords,
        )
        root_cubes = self.v2v_net(initial_cubes)
        root_cubes = root_cubes.squeeze(1)
        grid_centers = self.proposal_layer(root_cubes, meta, grids)

        return root_cubes, grid_centers

    def forward(self, all_heatmaps, meta, flip_xcoords=None):
        all_heatmaps_copy = [
            a[:, self.root_id, :, :][:, None].clone() for a in all_heatmaps
        ]

        initial_cubes, grids = self.project_layer(
            all_heatmaps_copy,
            meta,
            self.grid_size,
            [self.grid_center],
            self.cube_size,
            flip_xcoords=flip_xcoords,
        )

        root_cubes = self.v2v_net(initial_cubes)
        root_cubes = root_cubes.squeeze(1)
        grid_centers = self.proposal_layer(root_cubes, meta, grids)

        return root_cubes, None, None, grid_centers

