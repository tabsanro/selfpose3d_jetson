# built-in dependencies
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Dict, List, Optional, Union

# 3rd-party dependencies
import numpy as np
import torch

# SelfPose3d


# MiVOLO
from mivolo.structures import PersonAndFaceResult

def set_lod(r: float, roots: torch.Tensor):
    """
    Sets the level of detail (LOD) based on the given radius and root coordinates.

    Args:
        r (float): The radius used to determine the LOD.
        roots (torch.Tensor): The tensor containing the root coordinates.

    Returns:
        torch.Tensor: The tensor containing the LOD values for each root.
    """
    coords_2d = roots[:, :2]
    distances = torch.norm(coords_2d, dim=1)
    within_radius = distances <= r
    lod = torch.where(within_radius, torch.tensor(2, device=roots.device), torch.tensor(1, device=roots.device))
    return lod

def sync_objs(
        joints_3d:torch.Tensor,
        detected_objects:List[PersonAndFaceResult],
        camera_data:List[Dict[str, Any]],
        camera_num:int,
        device:torch.device
    ) -> List[Dict[str, int]]:
    """
    Synchronizes the detected objects with the 3D joints and camera data.

    Args:
        joints_3d (torch.Tensor): The 3D joints tensor.
        detected_objects (List[PersonAndFaceResult]): The list of detected objects.
        camera_data (List[Dict[str, Any]]): The list of camera data dictionaries.
        camera_num (int): The number of cameras.
        device (torch.device): The device to perform computations on.

    Returns:
        List[Dict[str, int]]: The synchronized person information.

    """
    process_func = partial(_process_detected_object, device=device)
    with ThreadPoolExecutor() as executor:
        processed_objects = list(executor.map(process_func, detected_objects))
    filters, bboxes, ages, genders, gender_scores = zip(*processed_objects)

    bboxes, ages, genders, gender_scores = _filter_tensor(filters, device, bboxes, ages, genders, gender_scores)

    joints_indices: List[List[int]] = [[] for _ in range(camera_num)]
    joints_as_much_as_cam = _project_joints(joints_3d, camera_data, device)

    for i, (joints_2d, bbox) in enumerate(zip(joints_as_much_as_cam, bboxes)):
        for person in joints_2d:
            joints_indices[i].append(sync_joints_bboxes(person, bbox))
        joints_indices[i] = resolve_joint_conflicts(joints_indices[i], joints_2d, bbox, device)

    return make_person_info(joints_indices, ages, genders, gender_scores)

def sync_joints_bboxes(joints: torch.Tensor, bboxes: torch.Tensor) -> int:
    """
    Synchronizes the joints and bounding boxes by finding the index of the bounding box
    that contains the maximum number of joints.

    Args:
        joints (torch.Tensor): A tensor containing the joint coordinates.
        bboxes (torch.Tensor): A tensor containing the bounding box coordinates.

    Returns:
        int: The index of the bounding box that contains the maximum number of joints.

    """

    if len(joints) == 0 or bboxes.size(0) == 0:
        return -1

    bboxes = bboxes.unsqueeze(1)
    joints = joints.unsqueeze(0)

    x1 = bboxes[:, :, 0].unsqueeze(-1)
    y1 = bboxes[:, :, 1].unsqueeze(-1)
    x2 = bboxes[:, :, 2].unsqueeze(-1)
    y2 = bboxes[:, :, 3].unsqueeze(-1)

    joints_x = joints[:, :, 0]
    joints_y = joints[:, :, 1]

    joints_in_bbox = (joints_x >= x1) & (joints_x <= x2) & (joints_y >= y1) & (joints_y <= y2)
    joint_counts = torch.sum(joints_in_bbox, dim=-1)

    max_joints_indices = torch.argmax(joint_counts, dim=0)
    return max_joints_indices.item()

def resolve_joint_conflicts(
        joint_bbox_indices: List[int],
        joints: List[List[List[int]]],
        bboxes: torch.Tensor,
        device:torch.device,
    ) -> List[int]:
    """
    Resolves conflicts between joints and bounding boxes by assigning the best joint index to each bounding box.

    Args:
        joint_bbox_indices (List[int]): A list of indices representing the assigned bounding box for each joint.
        joints (List[List[List[int]]]): A list of joint coordinates.
        bboxes (torch.Tensor): A tensor containing bounding box information.
        device (torch.device): The device on which the tensor is located.

    Returns:
        List[int]: A list of indices representing the assigned bounding box for each joint after resolving conflicts.

    """
    assigned_joints = [[] for _ in range(len(bboxes))]
    
    for joint_idx, bbox_idx in enumerate(joint_bbox_indices):
        if bbox_idx == -1:
            continue
        if bbox_idx is not None:
            assigned_joints[bbox_idx].append(joint_idx)
    
    for bbox_idx, joint_indices in enumerate(assigned_joints):
        if len(joint_indices) > 1:
            best_joint_idx = joint_indices[0]
            max_spread = 0
            
            for joint_idx in joint_indices:
                spread = _spread_extent(joints[joint_idx]).item()
                if spread > max_spread:
                    best_joint_idx = joint_idx
                    max_spread = spread
            
            for joint_idx in joint_indices:
                if joint_idx != best_joint_idx:
                    joint_bbox_indices[joint_idx] = None
    
    return joint_bbox_indices

def make_person_info(
    joint_indices: List[List[int]],
    ages: List[torch.Tensor],
    genders: List[torch.Tensor],
    gender_scores: List[torch.Tensor],
) -> List[Dict[str, int]]:
    """Creates a list of person information based on the synchronized joint indices, ages, genders, and gender scores.

    Args:
        joint_indices (List[List[int]]): The synchronized joint indices for each person.
        ages (List[torch.Tensor]): The ages of each person.
        genders (List[torch.Tensor]): The genders of each person.
        gender_scores (List[torch.Tensor]): The gender scores of each person.

    Returns:
        List[Dict[str, int]]: A list of dictionaries containing the person information, including age and gender.

    """
    person_len = max(len(joint_indices) for joint_indices in joint_indices)
    person_info = [{} for _ in range(person_len)]
    age_list: List[List[torch.Tensor]] = [[] for _ in range(person_len)]
    gender_list: List[List[torch.Tensor]] = [[] for _ in range(person_len)]
    score_list: List[List[torch.Tensor]] = [[] for _ in range(person_len)]

    for joint_indice, age, gender, score in zip(joint_indices, ages, genders, gender_scores):
        for idx in joint_indice:
            if idx is not None:
                try:
                    age_list[idx].append(age)
                    gender_list[idx].append(gender)
                    score_list[idx].append(score)
                except IndexError as e:
                    print(e)

    for idx in range(person_len):
        person_info[idx]['age'] = _trimmed_mean_age(age_list[idx])
        person_info[idx]['gender'] = _weighted_mean_gender(gender_list[idx], score_list[idx])

    return person_info

def _gender_to_int(data_list: List[Optional[str]]) -> List[Optional[int]]:
    """
    Converts a list of gender strings to a list of corresponding integer values.

    Args:
        data_list (List[Optional[str]]): A list of gender strings.

    Returns:
        List[Optional[int]]: A list of corresponding integer values, where 'male' is represented as 1,
        'female' is represented as 0, and other values are represented as None.

    """
    result: List[Optional[int]] = []
    for data in data_list:
        if data == 'male':
            result.append(1)
            continue
        if data == 'female':
            result.append(0)
            continue
        result.append(None)
    return result

def _spread_extent(joints:List[List[int]]) -> float:
    """
    Calculate the spread extent of the given joints.

    Args:
        joints (List[List[int]]): A list of joint coordinates.

    Returns:
        float: The spread extent of the joints.

    """
    x_range = torch.max(joints[:,0]) - torch.min(joints[:,0])
    y_range = torch.max(joints[:,1]) - torch.min(joints[:,1])
    return x_range * y_range

def _list_to_tensor(data_list: list, device: torch.device):
    """
    Converts a list to a tensor.

    Args:
        data_list (list): The input list to be converted to a tensor.
        device (torch.device): The device on which the tensor should be created.

    Returns:
        torch.Tensor: The tensor created from the input list.

    """
    return torch.tensor([0 if value is None else value for value in data_list], device=device)

def _filter_tensor(filters: List[Union[torch.Tensor, np.ndarray]], device: torch.device, *tensors: torch.Tensor) -> List[torch.Tensor]:
    """
    Filters the given tensors based on the provided filters.

    Args:
        filters (List[Union[torch.Tensor, np.ndarray]]): A list of filters to apply to the tensors.
        device (torch.device): The device to use for the filtered tensors.
        *tensors (torch.Tensor): The tensors to be filtered.

    Returns:
        List[torch.Tensor]: A list of filtered tensors.

    """
    filtered_tensors = []

    for tensor in tensors:
        filtered_list = []
        for tensor, filter in zip(tensor, filters):
            if filter.numel() == 0:
                filtered_list.append(tensor)
            else:
                mask = (filter == 0)
                filtered_list.append(tensor[mask])
        filtered_tensors.append(filtered_list)

    return tuple(filtered_tensors)

def _trimmed_mean_age(data: List[torch.Tensor], len_to_cut: int = 0) -> float:
    """
    Calculate the trimmed mean age of the given data.

    Args:
        data (List[torch.Tensor]): A list of tensors containing age data.
        len_to_cut (int, optional): The number of elements to trim from both ends of the sorted data. Defaults to 0.

    Returns:
        float: The trimmed mean age.

    """
    if not data:
        return 0

    max_len = max(tensor.size(0) for tensor in data)
    min_len = min(tensor.size(0) for tensor in data)

    if min_len <= 2:
        len_to_cut = 0

    padded_data = [
        torch.cat([tensor, torch.full((max_len - tensor.size(0),), float('nan'), device=tensor.device)])
        if tensor.size(0) < max_len else tensor
        for tensor in data
    ]

    data_tensor = torch.stack(padded_data)
    data_sorted, _ = torch.sort(data_tensor, dim=0)
    trimmed_data = data_sorted[len_to_cut:max_len - len_to_cut]
    mean_age = torch.nanmean(trimmed_data, dim=0)
    final_mean_age = mean_age[~torch.isnan(mean_age)].mean().item()

    return final_mean_age if not torch.isnan(torch.tensor(final_mean_age)) else 0

def _weighted_mean_gender(data:List[torch.Tensor], weight:list) -> int:
    """
    Calculate the weighted mean gender based on the given data and weights.

    Args:
        data (List[torch.Tensor]): A list of tensors representing the data.
        weight (list): A list of weights corresponding to the data.

    Returns:
        int: The weighted mean gender, where 1 represents male and 0 represents female.

    """
    if not data or not weight:
        return 1
    max_len = max(tensor.size(0) for tensor in data)
    padded_data = [torch.cat([tensor, torch.full((max_len - tensor.size(0),), float('nan'), device=tensor.device)]) if tensor.size(0) < max_len else tensor for tensor in data]
    padded_weights = [torch.cat([tensor, torch.full((max_len - tensor.size(0),), float('nan'), device=tensor.device)]) if tensor.size(0) < max_len else tensor for tensor in weight]
    data_tensor = torch.stack(padded_data)
    weight_tensor = torch.stack(padded_weights)
    weighted_mean = torch.nansum(data_tensor*weight_tensor) / torch.nansum(weight_tensor)
    return 1 if weighted_mean >= 0.5 else 0

def _process_detected_object(detected_object:PersonAndFaceResult, device:torch.device):
    """
    Process the detected object and extract relevant information.

    Args:
        detected_object (PersonAndFaceResult): The detected object containing information about the person and face.
        device (torch.device): The device to perform the processing on.

    Returns:
        tuple: A tuple containing the following information:
            object_classes (torch.Tensor): The classes of the detected objects.
            xyxy (torch.Tensor): The bounding box coordinates of the detected objects.
            ages (torch.Tensor): The ages of the detected persons.
            genders (torch.Tensor): The genders of the detected persons.
            gender_scores (torch.Tensor): The confidence scores for the detected genders.
    """
    yolo_results = detected_object.yolo_results.boxes
    object_classes = yolo_results.cls
    xyxy = yolo_results.xyxy
    ages = _list_to_tensor(detected_object.ages, device=device)
    genders = _list_to_tensor(_gender_to_int(detected_object.genders), device=device)
    gender_scores = _list_to_tensor(detected_object.gender_scores, device=device)
    return object_classes, xyxy, ages, genders, gender_scores

def _project_joints(joints_3d:torch.Tensor, camera_data:List[Dict[str, Any]], device:torch.device):
    """
    Projects 3D joint coordinates onto 2D image coordinates for multiple cameras.

    Args:
        joints_3d (torch.Tensor): Tensor containing 3D joint coordinates of shape (person_num, joints_num, 3).
        camera_data (List[Dict[str, Any]]): List of dictionaries containing camera parameters for each camera.
        device (torch.device): Device to perform the computation on.

    Returns:
        torch.Tensor: Tensor containing 2D joint coordinates of shape (cam_num, person_num, joints_num, 2).

    """

    joints_3d = joints_3d[..., :3] / 1000
    person_num, joints_num, _ = joints_3d.shape
    cam_num = len(camera_data)
    joints_2d = torch.zeros(cam_num, person_num, joints_num, 2, device=device)  # Initialize the 2D joints tensor

    for i in range(person_num):
        for j in range(cam_num):
            camera = camera_data[j]
            K = np.array([
                [camera['fx'], 0, camera['cx']],
                [0, camera['fy'], camera['cy']],
                [0, 0, 1]
            ])
            distCoef = np.zeros(5)
            distCoef[[0, 1, 4]] = camera['k'].flatten()
            distCoef[2:4] = camera['p'].flatten()
            joints_2d[j, i] = torch.from_numpy(projectPoints(
                joints_3d[i].transpose(0, 1).detach().cpu().numpy(), # Transpose to match the expected input shape
                K,
                camera['R'],
                camera['T'],
                distCoef,
            )).to(device).transpose(0, 1)[:, :2] # Transpose back to match the expected output shape

    return joints_2d

def projectPoints(X, K, R, t, Kd):
    """
    Projects points X (3xN) using camera intrinsics K (3x3),
    extrinsics (R,t) and distortion parameters Kd=[k1,k2,p1,p2,k3].
    Roughly, x = K*(R*X + t) + distortion
    See http://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
    or cv2.projectPoints
    """

    x = np.dot(R, X) + t

    x[0:2, :] = x[0:2, :] / (x[2, :] + 1e-5)

    r = x[0, :] * x[0, :] + x[1, :] * x[1, :]

    x[0, :] = x[0, :] * (1 + Kd[0] * r + Kd[1] * r * r + Kd[4] * r * r * r
                        ) + 2 * Kd[2] * x[0, :] * x[1, :] + Kd[3] * (
                            r + 2 * x[0, :] * x[0, :])
    x[1, :] = x[1, :] * (1 + Kd[0] * r + Kd[1] * r * r + Kd[4] * r * r * r
                        ) + 2 * Kd[3] * x[0, :] * x[1, :] + Kd[2] * (
                            r + 2 * x[1, :] * x[1, :])

    x[0, :] = K[0, 0] * x[0, :] + K[0, 1] * x[1, :] + K[0, 2]
    x[1, :] = K[1, 0] * x[0, :] + K[1, 1] * x[1, :] + K[1, 2]

    return x