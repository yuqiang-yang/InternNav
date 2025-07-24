import os

import numpy as np
import torch
from gym import spaces

from internnav.utils.common_log_util import common_logger as logger

from .basemodel.cma.cma_clip_policy import CMA_CLIP_Net, CMACLIPModelConfig
from .basemodel.cma.cma_policy import CMAModelConfig, CMANet
from .basemodel.rdp.rdp_policy import RDPNet, RDPModelConfig
from .basemodel.navdp.navdp_policy import NavDPModelConfig, NavDPNet

from .basemodel.seq2seq.seq2seq_policy import Seq2SeqNet, Seq2SeqModelConfig

from .basemodel.internvla_n1.internvla_n1_policy import InternVLAN1Net, InternVLAN1ModelConfig

from .utils.misc import set_cuda, set_random_seed, wrap_model
from .utils.save import load_checkpoint



def get_policy(policy_name):
    if policy_name == 'CMA_CLIP_Policy':
        return CMA_CLIP_Net
    elif policy_name == 'RDP_Policy':
        return RDPNet
    elif policy_name == 'CMA_Policy':
        return CMANet
    elif policy_name == 'Seq2Seq_Policy':
        return Seq2SeqNet
    elif policy_name == 'InternVLAN1_Policy':
        return InternVLAN1Net
    elif policy_name == 'NavDP_Policy':
        return NavDPNet
    else:
        raise ValueError(f'Policy {policy_name} not found')


def get_config(policy_name):
    if policy_name == 'CMA_CLIP_Policy':
        return CMACLIPModelConfig
    elif policy_name == 'RDP_Policy':
        return RDPModelConfig
    elif policy_name == 'CMA_Policy':
        return CMAModelConfig
    elif policy_name == 'Seq2Seq_Policy':
        return Seq2SeqModelConfig
    elif policy_name == 'InternVLAN1_Policy':
        return InternVLAN1ModelConfig
    elif policy_name == 'NavDP_Policy':
        return NavDPModelConfig
    else:
        raise ValueError(f'Policy {policy_name} not found')


def initialize_policy(
    config,
    device=None,
    action_stats=None,
) -> None:
    from internnav.utils.common_log_util import common_logger as logger  # 延迟导入

    load_from_ckpt = config.il.load_from_ckpt
    load_from_pretrain = config.il.load_from_pretrain

    default_gpu, n_gpu, device = set_cuda(config, device)
    if default_gpu:
        logger.info(
            'device: {} n_gpu: {}, distributed training: {}'.format(device, n_gpu, bool(config.local_rank != -1))
        )

    seed = config.seed
    if config.ddp.use:
        seed += config.local_rank
    set_random_seed(seed)

    # if default_gpu:
    #     save_training_meta(config)

    observation_space = spaces.Box(
        low=0.0,
        high=1.0,
        shape=(256, 256, 1),
        dtype=np.float32,
    )

    policy = get_policy(config.model.policy_name)

    self_policy = policy(
        config=config.model,
        observation_space=observation_space,
    )

    if load_from_pretrain:
        new_ckpt_weights = {}
        model_config = config.model
        self_policy.load_state_dict(new_ckpt_weights, strict=False)

    start_epoch = 0
    if load_from_ckpt:
        ckpt_path = config.il.ckpt_to_load
        ckpt_dict = load_checkpoint(ckpt_path, map_location='cpu')
        if 'state_dict' in ckpt_dict:
            state_dict = ckpt_dict['state_dict']
        else:
            state_dict = ckpt_dict
        if 'epoch' in ckpt_dict:
            start_epoch = ckpt_dict['epoch']
        new_state_dict = {}
        # Iterate through the state dictionary items
        for k, v in state_dict.items():
            new_key = k.replace('module.', '')
            if config.model.policy_name != 'RDP_Policy':
                new_key = new_key.replace('net.', '')  # this is for cma policy
            new_state_dict[new_key] = v
        del state_dict[k]  # Remove the old key with 'module.'

        incompatible_keys, _ = self_policy.load_state_dict(new_state_dict, strict=False)
        if len(incompatible_keys) > 0:
            logger.warning(f'Incompatible keys: {incompatible_keys}')
        logger.info(f'Loaded weights from checkpoint: {ckpt_path}')

    params = sum(param.numel() for param in self_policy.parameters())
    params_t = sum(p.numel() for p in self_policy.parameters() if p.requires_grad)
    logger.info(f'Agent parameters: {params / 1e6:.2f}M. Trainable: {params_t / 1e6:.2f}M')
    logger.info('Finished setting up policy.')

    if len(config.torch_gpu_ids) == 1:
        config.ddp.use = False
    if config.ddp.use:
        if config.ddp.use_dp:
            # Data parallel
            self_policy = wrap_model(
                self_policy,
                config.torch_gpu_ids,
                config.local_rank,
                logger,
                config.world_size,
                use_dp=config.ddp.use_dp,
            )
        else:
            # Distributed data parallel
            self_policy = wrap_model(
                self_policy,
                torch.device(f'cuda:{config.local_rank}'),
                config.local_rank,
                logger,
                config.world_size,
                use_dp=config.ddp.use_dp,
            )
    else:
        self_policy.to(device)

    optimizer = torch.optim.Adam(self_policy.parameters(), lr=float(config.il.lr))

    lr_scheduler = None
    if config.il.lr_schedule.use:
        if config.il.lr_schedule.type == 'cosine':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.il.epochs,
                eta_min=float(config.il.lr_schedule.min_lr),
            )
        elif config.il.lr_schedule.type == 'step':
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=config.il.lr_schedule.decay_epochs,
                gamma=float(config.il.lr_schedule.decay_factor),
            )
        elif config.il.lr_schedule.type == 'linear':
            lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=config.il.lr_schedule.warmup_factor,
                end_factor=1.0,
                total_iters=config.il.lr_schedule.warmup_epochs,
            )

    return self_policy, optimizer, lr_scheduler, start_epoch
