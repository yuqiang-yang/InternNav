import base64
import pickle

from internnav.evaluator.utils.common import load_scene_usd
from internnav.projects.dataloader.resumable import ResumablePathKeyDataloader
from internnav.projects.internutopia_vln_extension.configs.tasks.vln_eval_task import (
    VLNEvalTaskCfg,
)
from internnav.configs.evaluator import EvalCfg
from internutopia_extension.configs.robots.h1 import H1RobotCfg
from internutopia.core.config.robot import ControllerCfg
from internnav.projects.internutopia_vln_extension.configs.metrics.vln_pe_metrics import (
    VLNPEMetricCfg,
)
from internutopia_extension.configs.sensors import RepCameraCfg


def generate_episode(dataloader: ResumablePathKeyDataloader, config: EvalCfg):
    mp3d_data_dir = config.task.scene.mp3d_data_dir
    scene_asset_path = config.task.scene.scene_asset_path
    eval_path_key_list = dataloader.resumed_path_key_list
    path_key_data = dataloader.path_key_data
    episodes = []
    
    robot = H1RobotCfg(
        **config.task.robot.robot_settings,
        controllers=[ControllerCfg(**cfg.controller_settings) for cfg in config.task.robot.controllers],
        sensors=[RepCameraCfg(**cfg.sensor_settings) for cfg in config.task.robot.sensors],
    )

    for path_key in eval_path_key_list:
        data = path_key_data[path_key]
        start_position = data['start_position']
        start_rotation = data['start_rotation']
        data['path_key'] = path_key
        data['name'] = dataloader.task_name
        robot_flash = getattr(config.task, "robot_flash", False )
        episodes.append(
            VLNEvalTaskCfg(
                **config.task.task_settings,
                robot_flash=robot_flash,
                metrics=[VLNPEMetricCfg(**config.task.metric.metric_setting['metric_config'])],
                scene_asset_path=load_scene_usd(mp3d_data_dir, dataloader.path_key_scan[path_key])
                if scene_asset_path == ''
                else scene_asset_path,
                scene_scale=(1, 1, 1),
                robots=[
                    robot.update(
                        position=(
                            start_position[0],
                            start_position[1],
                            start_position[2],
                        ),
                        orientation=(
                            start_rotation[0],
                            start_rotation[1],
                            start_rotation[2],
                            start_rotation[3],
                        ),
                    )
                ],
                data=data,
            )
        )
    return episodes


def serialize_obs(obs):
    serialized = pickle.dumps(obs)
    encoded = base64.b64encode(serialized).decode('utf-8')
    return encoded
