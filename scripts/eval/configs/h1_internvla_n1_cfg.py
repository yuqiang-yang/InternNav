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
            'qwen_path': '/nav-oss/yangyuqiang/Qwen2.5-VL-7B-Instruct',
            'model_path': "/nav-oss/zhangsiqi/qwen_latent_ckpt_pjq34812/",
            'camera_intrinsic': [
                [585.0, 0.0, 320.0], [0.0, 585.0, 240.0], [0.0, 0.0, 1.0]
            ],
            
            'width': 640, 'height': 480, 'hfov': 79,
            'resize_w': 384, 'resize_h': 384,
            'max_new_tokens': 1024,
            'num_frames': 32,
            'num_history': 8,
            'num_future_steps': 4,
            
            'device': 'cuda:0',
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
            mp3d_data_dir='/shared/smartbot/vln-pe/Matterport3D/data/v1/scans',
        ),
        robot_name='h1',
        robot_flash=True, 
        robot_usd_path='/robots/h1/h1_vln_multi_camera.usd',
        camera_resolution=[640, 480] # (W,H)
    ),
    dataset=EvalDatasetCfg(
        dataset_settings={
            'base_data_dir': '/shared/smartbot/vln-pe/data/datasets/R2R_VLNCE_v1-3_corrected',
            'split_data_types': ['val_unseen'],  # 'val_seen'
            'filter_stairs': True,      
            # 'selected_scans': ['zsNo4HB9uLZ'],
            # 'selected_scans': ['8194nk5LbLH', 'pLe4wQe7qrG'],
        },
    ),
)
