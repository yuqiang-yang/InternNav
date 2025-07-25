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
        server_port=8080,
        model_name='seq2seq',
        ckpt_path='data/checkpoints/seq2seq/checkpoint-387159',
        model_settings={},
    ),
    env=EnvCfg(
        env_type='vln_pe',
        env_settings={
            'use_fabric': False,
            'headless': True,
        },
    ),
    task=TaskCfg(
        task_name='seq2seq_eval',
        task_settings={
            'env_num': 2,
            'use_distributed': True,
            'proc_num': 1,
        },
        scene=SceneCfg(
            scene_type='mp3d',
            mp3d_data_dir='/shared/smartbot/vln-pe/Matterport3D/data/v1/scans',
        ),
        robot_name='h1',
        robot_usd_path='/robots/h1/h1_vln_pointcloud.usd',
        camera_resolution=[256,256], # (W,H)
        camera_prim_path='torso_link/h1_pano_camera_0',
    ),
    dataset=EvalDatasetCfg(
        dataset_settings={
            'base_data_dir': 'data/datasets/R2R_VLNCE_v1-3_corrected',
            'split_data_types': ['val_unseen', 'val_seen'],
            'filter_stairs': True,
        },
    ),
)
