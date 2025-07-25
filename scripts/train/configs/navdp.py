from internnav.configs.model.navdp import navdp_cfg
from internnav.configs.trainer.eval import EvalCfg
from internnav.configs.trainer.exp import ExpCfg
from internnav.configs.trainer.il import FilterFailure, IlCfg, Loss
import os

navdp_exp_cfg = ExpCfg(
    name='navdp_train',
    model_name='navdp',
    # num_gpus = 4,
    torch_gpu_id=0,
    torch_gpu_ids=[0],
    output_dir='data/checkpoints/%s/ckpts',
    tensorboard_dir='data/checkpoints/%s/tensorboard',
    checkpoint_folder='data/checkpoints/%s/ckpts',
    log_dir='data/checkpoints/%s/logs',
    local_rank= 0,
    # device = None,
    seed=0,
    eval=EvalCfg(
        use_ckpt_config=False,
        save_results=True,
        split=['val_seen'],
        ckpt_to_load='',
        max_steps=195,
        sample=False,
        success_distance=3.0,
        start_eval_epoch=-1,
        step_interval=50,
    ),
    il=IlCfg(
        epochs=1000,
        batch_size=16,
        lr=1e-4,
        num_workers=8,
        weight_decay=1e-4,  # TODO
        warmup_ratio=0.05,  # TODO
        use_iw=True,
        inflection_weight_coef=3.2,
        save_interval_epochs=5,
        save_filter_frozen_weights=False,
        load_from_ckpt=False,
        ckpt_to_load='',
        load_from_pretrain=False,
        lmdb_map_size=1e12,
        dataset_r2r_root_dir='data/datasets/R2R_VLNCE_v1-3_preprocessed',
        dataset_3dgs_root_dir='data/datasets/3dgs',
        dataset_grutopia10_root_dir='data/datasets/grutopia10',
        lmdb_features_dir='data/sample_episodes/20250211_sample_origin/sample_data.lmdb',
        camera_name='pano_camera_0',
        report_to='tensorboard',  # wandb, tensorboard, none
        dataset_navdp = '/path/to/dataloader/multiview_dataset_hssd_modified.json',
        image_size=224,
        scene_scale=1.0,
        preload=True,
        random_digit=False,
        prior_sample=False,
        memory_size=8,
        predict_size=24,
        temporal_depth=16,
        heads=8,
        token_dim=384,
        channels=3,
        dropout=0.1,
        scratch=False,
        finetune=True,
        ddp_find_unused_parameters = True,
        filter_failure=FilterFailure(
            use=True,
            min_rgb_nums=15,
        ),
        loss=Loss(
            alpha=0.0001,
            dist_scale=1,
        ),
    ),
    model=navdp_cfg,
)
