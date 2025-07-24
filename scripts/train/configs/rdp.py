from internnav.configs.model.rdp import rdp_cfg
from internnav.configs.trainer.eval import EvalCfg
from internnav.configs.trainer.exp import ExpCfg
from internnav.configs.trainer.il import FilterFailure, IlCfg, Loss

rdp_exp_cfg = ExpCfg(
    name='rdp_train',
    model_name='rdp',
    torch_gpu_id=0,
    torch_gpu_ids=[0, 1, 2, 3],
    output_dir='data/checkpoints/%s/ckpts',
    checkpoint_folder='data/checkpoints/%s/ckpts',
    tensorboard_dir='data/checkpoints/%s/tensorboard',
    log_dir='data/checkpoints/%s/logs',
    seed=0,
    eval=EvalCfg(
        use_ckpt_config=False,
        save_results=True,
        split=['val_unseen'],
        max_steps=195,
        action='discrete',
        sample=True,
        success_distance=3.0,
        rotation_threshold=1e-2,
        num_sample=1,
        start_eval_epoch=-1,
        stop_mode='stop_progress',
        pm_threshold=0.90,
        step_interval=80,
        len_traj_act=2,
    ),
    il=IlCfg(
        epochs=50,
        save_interval_epochs=5,
        batch_size=2,
        lr=1e-4,
        num_workers=8,
        weight_decay=1e-4,
        warmup_ratio=0.1,
        save_filter_frozen_weights=True,
        load_from_ckpt=False,
        ckpt_to_load='',
        load_from_pretrain=True,
        dataset_r2r_root_dir='data/datasets/R2R_VLNCE_v1-3_corrected',
        dataset_3dgs_root_dir='data/datasets/3dgs',
        dataset_grutopia10_root_dir='data/datasets/grutopia10',
        lmdb_features_dir='r2r',
        lerobot_features_dir='data/datasets/vln_pe_lerobot/mp3d',
        camera_name='pano_camera_0',
        report_to='wandb',  # wandb, tensorboard, none
        ddp_find_unused_parameters = True,
        filter_failure=FilterFailure(
            use=True,
            min_rgb_nums=15,
        ),
        use_discrete_dataset=True,
        loss=Loss(
            alpha=0.0001,
            dist_scale=1,
        ),
    ),
    model=rdp_cfg,
)
