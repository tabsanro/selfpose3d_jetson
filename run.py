import os
import argparse
from typing import List, Tuple, Union

import torch
import torchvision.transforms as transforms
from tqdm import tqdm

from configs.config import config as focus_config
from configs.config import update_config as update_focus_config
from modules.focus.dataset import FOCUSDataset as dataset
from modules.SelfPose3d.lib.models.multi_person_posenet_ssv import get_multi_person_pose_net
from modules.SelfPose3d.lib.core.config import config as sp3d_config
from modules.SelfPose3d.lib.core.config import update_config as update_sp3d_config

CWD = os.getcwd()

def get_parser():
    parser = argparse.ArgumentParser(description="PyTorch AISL Inference")
    parser.add_argument("--cfg_focus", default='configs/focus.yaml', help="experiment configure file name", type=str)
    parser.add_argument("--cfg_sp3d", default='modules/SelfPose3d/config/cam4_posenet.yaml', help="experiment configure file name", type=str)
    parser.add_argument("--source_folder", default=None, help="source folder name", type=str)
    parser.add_argument("--webcam", type=bool, default=False, help="If set, the program will use webcam.")
    args, rest = parser.parse_known_args()
    if args.webcam:
        parser.add_argument("--webcam_info", type=str, default=None, help="Webcam info file path.")
    return parser

# Factory method
def get_sources_and_calibs(cfg) -> Union[Tuple[List[str], str], Tuple[None, None]]:
    if cfg.WEBCAM:
        return None, None
    elif cfg.SOURCE_FOLDER_NAME is None:
        sources = [os.path.join('modules', 'SelfPose3d', 'data_0705', 'hdVideos', 'hd_00_{:02d}.mp4'.format(i+1)) for i in range(cfg.NUM_SOURCES)]
        calib_path = os.path.join('modules', 'SelfPose3d', 'data_0705', 'calibration')
        return sources, calib_path
    else:
        sources = [os.path.join('data', 'focus-dataset', 'data', cfg.SOURCE_FOLDER_NAME, 'hdVideos', 'hd_00_{:02d}.mp4'.format(i+1)) for i in range(cfg.NUM_SOURCES)]
        calib_path = os.path.join('data', 'focus-dataset', 'data', cfg.SOURCE_FOLDER_NAME, 'calibration')
        return sources, calib_path

def zero_padding(image, target_size=(224, 224)):
    # 특이하게 tensor 주제에 np마냥 shape이 (h, w, c)로 나옴
    # 보통 tensor는 (c, h, w)로 나옴
    # 그래서 순서 바꿀거임
    # image = image.permute(2, 0, 1)
    image = torch.from_numpy(image).cuda().permute(2, 0, 1)
    
    _, h, w = image.shape
    target_h, target_w = target_size
    
    pad_h = target_h - h
    pad_w = target_w - w
    
    pad_top = pad_h // 2 if pad_h > 0 else 0
    pad_bottom = pad_h - pad_top if pad_h > 0 else 0
    pad_left = pad_w // 2 if pad_w > 0 else 0
    pad_right = pad_w - pad_left if pad_w > 0 else 0
    
    padding = (pad_left, pad_right, pad_top, pad_bottom)
    
    padded_image = torch.nn.functional.pad(image, padding, mode='constant', value=0)
    
    return padded_image

def post_process(preds_3d, grid_centers, face_images):
    # POSE
    # -1 is not person
    # 0 is in lod2
    # 1 is in lod1
    # we predict always total 10 persons
    lod_list = []
    preds_3d = preds_3d[preds_3d[...,3]!=-1]  
    grid_centers = grid_centers[grid_centers[...,3]!=-1]
    # in_lod = preds_3d[preds_3d[...,3]==0].view(-1, 15, 5)
    in_lod = grid_centers[grid_centers[...,3]==0].view(-1, 5)
    num_person = in_lod.shape[0]
    for value in grid_centers[...,3]:
        if value.item() == 0:
            lod_list.append(2)
        else:
            lod_list.append(1)
    
    if preds_3d.size(0) == 0:
        preds_3d = None # No person detected in lod2
    else:
        preds_3d = preds_3d.squeeze(0).view(-1, 15, 5).detach().cpu().numpy()  # (batch, n, 15, 5)
    # No person detected in lod1
    if grid_centers.size(0) == 0:
        grid_centers = None
    else:
        grid_centers = grid_centers.squeeze(0).view(-1, 5).detach().cpu().numpy()

    # FACE
    face_images_tensor = []
    for n in range(len(face_images)):   # n: persons
        person_tensors = []
        for b in range(len(face_images[n])):    # b: batch
            batch_tensors = []
            for c in range(len(face_images[n][b])):   # c: cameras
                padded_img = zero_padding(face_images[n][b][c])
                batch_tensors.append(padded_img)
            person_tensors.append(torch.stack(batch_tensors).cuda())
        face_images_tensor.append(torch.stack(person_tensors).cuda())
    
    if face_images_tensor != []:
        face_images_tensor = torch.stack(face_images_tensor).cuda()
        face_images_tensor = face_images_tensor.squeeze(1)  # delete batch dim
    
    return preds_3d, grid_centers, face_images_tensor, num_person, lod_list

def load_model(model, ckpt_path):
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    model.module.load_state_dict(torch.load(ckpt_path), strict=False)
    model.eval()
    return model

def transform_image(image):
    image = image.float()/255.0
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image)

def edge_inference(model, inputs, num_person):
    # edge variables
    w_set = [1, 1, 1]
    nof = 2
    cof = 1
    tof = 1
    # 0: 알고리즘에 대한 시간과 Partition point
    # 3: cloud에서의 시간과 Partition point
    # 4: edge에서의 시간과 Partition point
    Algoritm_type = 0
    final_index, final_time, partition_point, E_f_time, C_f_time, tr_f_time = time_point_calculator.sync_algorithm(Algoritm_type, num_person, w_set, nof, cof, tof)
    print(f'\nPartition Algorithm Print\n{0} {1} {2} {3} {4} {5}\n'.format(final_index, final_time, partition_point, E_f_time, C_f_time, tr_f_time))
    print(final_index, final_time, partition_point, E_f_time, C_f_time, tr_f_time)
    
    results = []
    for idx, img in enumerate(inputs):
        _, _, featuremap = model(
            transform_image(img),
            type='test_AFAD',
            sp=0,
            ep=partition_point[idx],
        )
        results.append((featuremap, partition_point[idx]))
    return results

@torch.no_grad()
def main():
    parser = get_parser()
    args = parser.parse_args()
    # Update config
    # focus_config_path = 'configs/focus.yaml'
    update_focus_config(args.cfg_focus)
    # sp3d_config_path = 'modules/SelfPose3d/config/cam4_posenet.yaml'
    update_sp3d_config(args.cfg_sp3d)
    # then you can use `focus_config` and `sp3d_config` as a global variable

    sources, calib_path = get_sources_and_calibs(focus_config)
    pipelines = None
    
    # Set CUDA device
    gpus = [0]
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.enabled = sp3d_config.CUDNN.ENABLED
        torch.backends.cudnn.benchmark = sp3d_config.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = sp3d_config.CUDNN.DETERMINISTIC
        device = torch.device('cuda')
    else:
        raise ValueError('CUDA is not available. Please check your configuration.')

    data_set = dataset(focus_config, sources=sources, calib_path=calib_path, pipelines=pipelines, device=device)
    data_loader = torch.utils.data.DataLoader(
        dataset=data_set,
        batch_size=1,
        shuffle=False,
    )
    
    # load model
    pose_model = load_model(
        get_multi_person_pose_net(
            sp3d_config,
            is_train=False,
            tensorrt=focus_config.MODEL.POSENET.TENSORRT,
            engine_path=os.path.join(CWD, 'modules', 'SelfPose3d', 'models', 'backbone.engine'),
        ),
        focus_config.MODEL.POSENET.CKPT,
    )

    # Inference
    for origin_frames, transed_frames, meta in (pbar := tqdm(data_loader)):
        # Update distance
        distance = None
        
        # Set Results
        results = [] # 사람 수 만큼 결과 저장
        # 사람 = {lod , root, pred, age, gender}

        # Pose Estimation
        pred_3d, _, _, roots, face_images = pose_model(
            raw_images=origin_frames,
            views1=transed_frames,
            meta1=meta,
            distance=distance,
        )
        pred_3d, roots, face_images, num_person, lod_list = post_process(pred_3d, roots, face_images)
        if roots is None:
            continue
        for num_roots in range(len(roots)):
        # for lod, pred, root in zip(lod_list, pred_3d, roots):
            temp_dict = {}
            temp_dict['lod'] = lod_list[num_roots]
            temp_dict['root'] = roots[num_roots]
            if pred_3d is not None:
                # WARNING: lod2인데 pose가 없는 경우도 있음
                # 어차피 영상이다보니 이런 경우는 출력을 하지 않아도 시계열 연속성이 있는 것 처럼 보임
                # 다음 프레임에서 pose를 딸 확률이 높기 때문에
                try:
                    temp_dict['pred'] = pred_3d[num_roots]
                except:
                    temp_dict['pred'] = None
            else:
                temp_dict['pred'] = None
            results.append(temp_dict)

if __name__ == '__main__':
    main()
