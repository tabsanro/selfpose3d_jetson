import torch

def crop_front_body_image(frames, front_cam_idx, pose_2d):
    front_body_image = []
    for i, frame in enumerate(frames):
        if front_cam_idx[i]:
            max_x = torch.max(pose_2d[:, 0])
            min_x = torch.min(pose_2d[:, 0])
            max_y = torch.max(pose_2d[:, 1])
            min_y = torch.min(pose_2d[:, 1])

            front_body_image.append(frame[min_y:max_y, min_x:max_x])

    return front_body_image