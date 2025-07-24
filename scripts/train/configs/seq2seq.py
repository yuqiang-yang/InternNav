from internnav.configs.model.seq2seq import seq2seq_cfg
from internnav.configs.trainer.eval import EvalCfg
from internnav.configs.trainer.exp import ExpCfg
from internnav.configs.trainer.il import FilterFailure, IlCfg, Loss

seq2seq_exp_cfg = ExpCfg(
    name='seq2seq_train',
    model_name='seq2seq',
    torch_gpu_id=0,
    torch_gpu_ids=[0],
    output_dir='data/checkpoints/%s/ckpts',
    checkpoint_folder='data/checkpoints/%s/ckpts',
    tensorboard_dir='data/checkpoints/%s/tensorboard',
    log_dir='data/checkpoints/%s/logs',
    seed=0,
    eval=EvalCfg(
        use_ckpt_config=False,
        save_results=True,
        split=['val_seen'],
        max_steps=195,
        sample=False,
        success_distance=3.0,
        rotation_threshold=1e-2,
        start_eval_epoch=-1,
        step_interval=50,
    ),
    il=IlCfg(
        epochs=80,
        save_interval_epochs=5,
        batch_size=2,
        lr=1e-4,
        num_workers=8,
        weight_decay=1e-4,
        warmup_ratio=0.1,
        use_iw=True,
        inflection_weight_coef=3.2,
        save_filter_frozen_weights=False,
        load_from_ckpt=False,
        ckpt_to_load='',
        load_from_pretrain=True,
        lmdb_map_size=1e12,
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
        loss=Loss(
            alpha=0.0001,
            dist_scale=1,
        ),
    ),
    model=seq2seq_cfg,
)
