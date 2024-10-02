from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.utils.data import DataLoader
import torch.utils.data.distributed

from tqdm import tqdm
import os
import os.path as osp
import pickle
import cv2

import _init_paths
from core.config import config
from core.config import update_config
from utils.visualize import display_3d, display_2d
import models
import dataset

this_dir = osp.dirname(osp.abspath(__file__))
# config 파일 설정을 불러오기 위한 함수
config_path = osp.join(this_dir, '..', 'config', 'cam4_posenet.yaml')
update_config(config_path)
# 사용할 사전 학습된 모델 경로
model_checkpoint = osp.join(this_dir, '..', 'models', 'POC_posenet.pth.tar')

gpus = [0]

# 모델 불러오기
# pose_model = models.load_model(config, model_checkpoint, gpus)
pose_model = models.load_model(config_path, model_checkpoint, gpus)

start_idx = 143   # 시작 프레임 인덱스
end_idx = None  # 끝 프레임 인덱스, None일 경우 마지막까지

test_dataset_cv2 = eval("dataset."+"POCdatasetCV2")(
    config,
    start_idx=start_idx, 
    end_idx=end_idx)

test_loader_cv2 = DataLoader(
        test_dataset_cv2,
        batch_size=1,
        shuffle=False)


pose_model.eval()
preds, preds_2d, roots, front_camera_group = [], [], [], []
with torch.no_grad():
    for i, (raw_images, inputs, meta) in enumerate(tqdm(test_loader_cv2)):
        pred, pred_2d, heatmaps, grid_centers, front_camera_indices, front_facing_images = pose_model(raw_images = raw_images, views1=inputs, meta1=meta, inference=True)

        pred = pred.detach().cpu().numpy()
        root = grid_centers.detach().cpu().numpy()
        pred_2d = pred_2d.detach().cpu().numpy()
        front_camera_indices = front_camera_indices.detach().cpu().numpy()
        for b in range(pred.shape[0]):
            preds.append(pred[b])
            preds_2d.append(pred_2d[b])
            roots.append(root[b])
            front_camera_group.append(front_camera_indices[b])
        for j, imgs in enumerate(front_facing_images):
            for k, img in enumerate(imgs[0]):
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"results/{i}th_{j}person_{k}view.jpg", img)

        


print("Inference done.")

# 결과 저장
output_dir = osp.join(this_dir, '..', 'output')
os.makedirs(output_dir, exist_ok=True)
output_file = osp.join(output_dir, 'output.pkl')
with open(output_file, 'wb') as f:
    pickle.dump({'preds': preds, 'preds_2d': preds_2d, 'roots': roots}, f)
    print(f"Results saved as {output_file}")


# 결과 시각화
# display_3d(preds=preds, output=output_file)
# display_2d(preds_2d=preds_2d, view_idx=1, output=output_file)