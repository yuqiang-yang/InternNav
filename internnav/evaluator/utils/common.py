import json
import math
import gzip
import copy
import os
from collections import defaultdict

import numpy as np
from internutopia.core.util import is_in_container
from scipy.ndimage import binary_dilation

from internnav.utils.common_log_util import common_logger as log


def create_robot_mask(topdown_global_map_camera, mask_size=20):
    height, width = topdown_global_map_camera._camera._resolution
    center_x, center_y = width // 2, height // 2
    # Calculate the top-left and bottom-right coordinates
    half_size = mask_size // 2
    top_left_x = center_x - half_size
    top_left_y = center_y - half_size
    bottom_right_x = center_x + half_size
    bottom_right_y = center_y + half_size

    # Create the mask
    robot_mask = np.zeros((width, height), dtype=np.uint8)
    robot_mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 1
    return robot_mask


def create_dilation_structure(voxel_size, radius):
    """
    Creates a dilation structure based on the robot's radius.
    """
    radius_cells = int(np.ceil(radius / voxel_size))
    # Create a structuring element for dilation (a disk of the robot's radius)
    dilation_structure = np.zeros((2 * radius_cells + 1, 2 * radius_cells + 1), dtype=bool)
    cy, cx = radius_cells, radius_cells
    for y in range(2 * radius_cells + 1):
        for x in range(2 * radius_cells + 1):
            if np.sqrt((x - cx) ** 2 + (y - cy) ** 2) <= radius_cells:
                dilation_structure[y, x] = True
    return dilation_structure


def freemap_to_accupancy_map(
    topdown_global_map_camera,
    freemap,
    dilation_iterations=0,
    voxel_size=0.1,
    agent_radius=0.25,
):
    height, width = topdown_global_map_camera._camera._resolution
    occupancy_map = np.zeros((width, height))
    occupancy_map[freemap == 1] = 2
    occupancy_map[freemap == 0] = 255
    if dilation_iterations > 0:
        dilation_structure = create_dilation_structure(voxel_size, agent_radius)
        for i in range(1, dilation_iterations):
            ob_mask = np.logical_and(occupancy_map != 0, occupancy_map != 2)
            expanded_ob_mask = binary_dilation(ob_mask, structure=dilation_structure, iterations=1)
            occupancy_map[expanded_ob_mask & (np.logical_or(occupancy_map == 0, occupancy_map == 2))] = 255 - i * 10
    return occupancy_map


def check_robot_fall(
    robot_position,
    robot_rotation,
    robots_bottom_z,
    pitch_threshold=35,
    roll_threshold=15,
    height_threshold=0.5,
):
    from omni.isaac.core.utils.rotations import quat_to_euler_angles

    roll, pitch, yaw = quat_to_euler_angles(robot_rotation, degrees=True)
    # Check if the pitch or roll exceeds the thresholds
    if abs(pitch) > pitch_threshold or abs(roll) > roll_threshold:
        is_fall = True
        log.debug('Robot falls down!!!')
        log.debug(f'Current Position: {robot_position}, Orientation: {roll, pitch, yaw}')
    else:
        is_fall = False

    # Check if the height between the robot base and the robot ankle is smaller than a threshold
    robot_ankle_z = robots_bottom_z
    robot_base_z = robot_position[2]
    if robot_base_z - robot_ankle_z < height_threshold:
        is_fall = True
        log.debug('Robot falls down!!!')
        log.debug(f'Current Position: {robot_position}, Orientation: {roll, pitch, yaw}')
    return is_fall


def describe_action(action):
    if action == 1:
        return '向前走0.25米'
    elif action == 2:
        return '左转15°'
    elif action == 3:
        return '右转15°'
    else:
        return '结束'


def get_action_state(obs, action_name):
    controllers = obs['controllers']
    action_state = controllers[action_name]['finished']
    return action_state


def check_is_on_track(
    robot_position,
    robot_rotation,
    action,
    action_index,
    real_points,
):
    if action == 1:
        distance = np.linalg.norm(robot_position[:2] - real_points[action_index][:2])
        if distance > 0.5:
            log.debug(f'[distance:{round(distance, 2)} > 0.5 ] replanning')
            return False
    else:
        from omni.isaac.core.utils.rotations import quat_to_euler_angles

        _, _, real_yaw = quat_to_euler_angles(robot_rotation)
        yaw_diff = abs(real_yaw - real_points[action_index])
        if yaw_diff > math.pi / 6:
            log.debug(f'[yaw_diff: {round(yaw_diff * (180 / math.pi))} 度 > 30 度] replanning')
            return False
    return True


def has_stairs(item, height_threshold=0.3):
    has_stairs = False
    if 'stair' in item['instruction']['instruction_text']:
        latest_height = item['reference_path'][0][-1]
        for index in range(1, len(item['reference_path'])):
            position = item['reference_path'][index]
            if abs(position[-1] - latest_height) >= height_threshold:
                has_stairs = True
                break
            else:
                latest_height = position[-1]
    return has_stairs


def different_height(item):
    different_height = False
    paths = item['reference_path']
    for path_idx in range(len(paths) - 1):
        if abs(paths[path_idx + 1][2] - paths[path_idx][2]) > 0.3:
            different_height = True
            break
    return different_height

def transform_rotation_z_90degrees(rotation):
    z_rot_90 = [np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)]  # 90 degrees = pi/2 radians
    w1, x1, y1, z1 = rotation
    w2, x2, y2, z2 = z_rot_90
    revised_rotation = [
        w1*w2 - x1*x2 - y1*y2 - z1*z2,  # w
        w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
        w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
        w1*z2 + x1*y2 - y1*x2 + z1*w2   # z
    ]
    return revised_rotation


def load_data(
    dataset_root_dir,
    split,
    filter_same_trajectory=True,
    filter_stairs=True
):
    with gzip.open(os.path.join(dataset_root_dir, split, f"{split}.json.gz"), 'rt', encoding='utf-8') as f:
        data = json.load(f)['episodes']
        
    scenes = list(set([x['scene_id'] for x in data]))  # e.g. 'mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb'
    new_data = {}
    for scene in scenes:
        scene_data = [x for x in data if x['scene_id'] == scene]
        scan = scene.split('/')[1]  # e.g. 'zsNo4HB9uLZ'
        new_scene_data = []
        for item in scene_data:
            new_item = copy.deepcopy(item)
            new_item['scan'] = scan
            new_item['original_start_position'] = item['start_position']
            new_item['original_start_rotation'] = item['start_rotation']
            x, z, y = item['start_position']
            new_item['start_position'] = [x, -y, z]
            r1, r2, r3, r4 = item['start_rotation']
            new_item['start_rotation'] = transform_rotation_z_90degrees([-r4, r1, r3, -r2])
            new_item['reference_path'] = [[x, -y, z] for x, z, y in item['reference_path']]
            new_scene_data.append(new_item)

        new_data[scan] = new_scene_data
        
    data = copy.deepcopy(new_data)
    new_data = defaultdict(list)
    
    # filter_same_trajectory
    if filter_same_trajectory:
        total_count = 0
        remaining_count = 0
        trajectory_list = []
        for scan, data_item in data.items():
            for item in data_item:
                total_count += 1
                if item['trajectory_id'] in trajectory_list:
                    continue
                remaining_count += 1
                trajectory_list.append(item['trajectory_id'])
                new_data[scan].append(item)
        log.info(f'[split:{split}]filter_same_trajectory remain: [ {remaining_count} / {total_count} ]')
        data = new_data
        new_data = defaultdict(list)

    if filter_stairs:
        total_count = 0
        remaining_count = 0
        for scan, data_item in data.items():
            for item in data_item:
                total_count += 1
                if has_stairs(item) or different_height(item):
                    continue
                remaining_count += 1
                new_data[scan].append(item)
        log.info(f'[split:{split}]filter_stairs remain: [ {remaining_count} / {total_count} ]')
        data = new_data

    return data


def load_scene_usd(mp3d_data_dir, scan):
    """Load scene USD based on the scan"""
    find_flag = False
    for root, dirs, files in os.walk(os.path.join(mp3d_data_dir, scan)):
        target_file_name = 'fixed_docker.usd' if is_in_container() else 'fixed.usd'
        for file in files:
            if file == target_file_name:
                scene_usd_path = os.path.join(root, file)
                find_flag = True
                break
        if find_flag:
            break
    if not find_flag:
        log.error('Scene USD not found for scan %s', scan)
        return None
    return scene_usd_path

def get_new_position_and_rotation(robot_position, robot_rotation, action):
    from omni.isaac.core.utils.rotations import (
        euler_angles_to_quat,
        quat_to_euler_angles,
    )

    roll, pitch, yaw = quat_to_euler_angles(robot_rotation)
    if action == 1:  # forward
        dx = 0.25 * math.cos(yaw)
        dy = 0.25 * math.sin(yaw)
        new_robot_position = robot_position + [dx, dy, 0]
        new_robot_rotation = robot_rotation
    elif action == 2:  # left
        new_robot_position = robot_position
        new_yaw = yaw + (math.pi / 12)
        new_robot_rotation = euler_angles_to_quat(np.array([roll, pitch, new_yaw]))
    elif action == 3:  # right
        new_robot_position = robot_position
        new_yaw = yaw - (math.pi / 12)
        new_robot_rotation = euler_angles_to_quat(np.array([roll, pitch, new_yaw]))
    else:
        new_robot_position = robot_position
        new_robot_rotation = robot_rotation
    return new_robot_position, new_robot_rotation


def set_seed(seed):
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    from omni.isaac.core.utils.torch.maths import set_seed

    set_seed(seed, torch_deterministic=True)
    import omni.isaac.core.utils.torch as torch_utils

    torch_utils.set_seed(seed)
    import omni.replicator.core as rep

    rep.set_global_seed(seed)


def set_seed_model(seed):
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False


def norm_depth(depth_info, min_depth=0, max_depth=10):
    depth_info[depth_info > max_depth] = max_depth
    depth_info = (depth_info - min_depth) / (max_depth - min_depth)
    return depth_info
