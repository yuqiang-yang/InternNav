# from scripts.eval.configs.agent import *
from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import (
    EnvCfg,
    EvalCfg,
    EvalDatasetCfg,
    SceneCfg,
    TaskCfg,
)

eval_cfg = EvalCfg(
    agent=AgentCfg(
        server_port=8023,
        model_name='internvla_n1',
        ckpt_path='',
        model_settings={
            'env_num': 1, 'sim_num': 1,
            'model_path': "checkpoints/InternVLA-N1",
            'camera_intrinsic': [
                [585.0, 0.0, 320.0], [0.0, 585.0, 240.0], [0.0, 0.0, 1.0]
            ],
            
            'width': 640, 'height': 480, 'hfov': 79,
            'resize_w': 384, 'resize_h': 384,
            'max_new_tokens': 1024,
            'num_frames': 32,
            'num_history': 8,
            'num_future_steps': 4,
            
            'device': 'cuda:1',
            'predict_step_nums': 32,
            'continuous_traj': True, 
            # debug
            'vis_debug': True, # If vis_debug=True, you can get visualization results
            'vis_debug_path': './logs/test/vis_debug' 
        },
    ),
    env=EnvCfg(
        env_type='vln_multi',
        env_settings={
            'use_fabric': False, # Please set use_fabric=False due to the render delay;
            'headless': True,
        },
    ),
    task=TaskCfg(
        task_name='test',
        task_settings={
            'env_num': 1,
            'use_distributed': False, # If the others setting in task_settings, please set use_distributed = False.
            'proc_num': 1,
        },
        scene=SceneCfg(
            scene_type='mp3d',
            mp3d_data_dir='data/scene_data/mp3d_pe',
        ),
        robot_name='h1',
        robot_flash=True, # If robot_flash is True, the mode is flash (set world_pose directly); else you choose physical mode.
        robot_usd_path='data/Embodiments/vln-pe/h1/h1_internvla.usd',
        camera_resolution=[640, 480], # (W,H)
        camera_prim_path='torso_link/h1_1_25_down_30',
        one_step_stand_still = True, #For dual-system, please keep this param True.
    ),
    dataset=EvalDatasetCfg(
        dataset_settings={
            'base_data_dir': 'data/vln_pe/raw_data',
            'split_data_types': ['val_unseen'],  # 'val_seen'
            'filter_stairs': True,      
            # 'selected_scans': ['zsNo4HB9uLZ'],
            # 'selected_scans': ['8194nk5LbLH', 'pLe4wQe7qrG'],
        },
    ),
)
