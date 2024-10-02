import argparse
from datetime import datetime
import os
import pickle

import cv2
import torch
from torchvision import transforms
from tqdm import tqdm

import _init_paths
from config import config, update_config
import focus as fc
import mivolo as mv
import SelfPose3d as sp3d

ROOT = os.path.dirname(os.path.abspath(__file__))

def get_parser():
    parser = argparse.ArgumentParser(description="PyTorch AISL Inference")
    parser.add_argument("--save-results", type=bool, default=True, help="If set, the results will be saved.")
    parser.add_argument("--cfg-focus", default="configs/focus.yaml", help="experiment configure file name", type=str)
    parser.add_argument("--verbose", action="store_true", help="If set, the program will print more information.")
    args, rest = parser.parse_known_args()
    update_config(args.cfg_focus)
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    # sources & calib_path
    # TODO if WEBCAM, real-time calib is required
    sources, calib_path = fc.get_sources_and_calibs(config)

    # if verbose flag is set, print the sources
    if args.verbose:
        if config.WEBCAM:
            print(f'Using webcam. {config.NUM_SOURCES} sources will be recorded.')
        else:
            print(f'Sources: {sources}')
    
    
    # gpus
    gpus = [0]
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        raise ValueError('CUDA is not available. Please check your configuration.')
    
    # if verbose flag is set, print the gpus
    if args.verbose:
        print(f'Using GPUs: {gpus}')

    # data loader
    data_set = fc.FOCUSDataset(config, sources, calib_path, device='cuda')
    data_loader = torch.utils.data.DataLoader(
        dataset=data_set,
        batch_size=1,
        shuffle=False,
        # pin_memory=True,
        # pin_memory_device='cuda',
    )
    # TODO preds_2d
    # camera_data will be deprecated because of preds_2d in the future
    camera_data = data_set.camera_data
    
    # if verbose flag is set, print the data loader
    if args.verbose:
        print(f'Data loader: {data_loader}')


    # models & visualizer
    # PoseModel
    sp3d_model = fc.PoseModel(gpus, ROOT, config)
    # AnalysisModel
    analysis_model = fc.get_analysis_model(config)
    # Visualizer
    visualizer = fc.get_visualizer(config)
    
    # if verbose flag is set, print the models
    if args.verbose:
        print(f'SelfPose3d model: {sp3d_model}')
        print(f'Analysis model: {analysis_model}')
        print(f'Visualizer: {visualizer}')
    
    
    # inference
    if args.save_results:
        os.makedirs('results', exist_ok=True)
        results = []
    try:
        with torch.no_grad():
            for origin_frames, trans_frames, meta in tqdm(data_loader):
                # Pose estimation
                preds_3d, preds_2d, _, roots, _ = sp3d_model(raw_images=origin_frames, views1=trans_frames, meta1=meta)
                # i want this but it's not working
                # if roots is None:
                #     continue
                if preds_3d is None:
                    continue
                
                # Analysis
                # TODO preds_2d로 대체, 디버그 해보니 preds_2d가 (40, 15,2)라서 사용 불가
                origin_frames = origin_frames.squeeze(0)    # (1, num_sources, H, W, C) -> (num_sources, H, W, C)
                frames = [frame.squeeze(0).numpy() for frame in origin_frames]
                obj_info = analysis_model.analyze(frames=frames, preds=preds_3d, roots=roots, camera_data=camera_data)

                # # 2D front directed pose image crop
                # front_body_images = []
                # for front_cam_idx_, pred_2d in zip(front_cam_idx, preds_2d):
                #     front_body_image = sp3d.crop_front_body_image(frames, front_cam_idx_, pred_2d)
                #     front_body_images.append(front_body_image)
                
                # Visualization
                visualizer.visualize(obj_info=obj_info)
                
                # Save results
                if args.save_results:
                    results.append(obj_info)
                
    except KeyboardInterrupt:
        print('KeyboardInterrupt')
        if not args.save_results:
            print('If you want to save the results, please run the program again with the --save-results flag.')
    
    
    # if save-results flag is set, save the results
    if args.save_results:
        today = datetime.today().strftime('%y%m%d_%H%M')
        with open(f'results/results{today}.pkl', 'wb') as f:
            pickle.dump(results, f)
    
    # if verbose flag is set, print the inference results
    if args.verbose:
        print(f'Inferenced {len(results)} frames.')
        if args.save_results:
            print(f'Results saved as results/results{today}.pkl')


if __name__ == '__main__':
    main()
