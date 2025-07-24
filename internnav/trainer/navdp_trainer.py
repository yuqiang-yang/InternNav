import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from internnav.trainer.base import BaseTrainer
import os
import time
from datetime import datetime
import multiprocessing

class NavDPTrainer(BaseTrainer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.writer = None
        # self.iterations = config.checkpoint * 141
        self.start_time = time.time()
        # 添加设备属性
        if hasattr(self.model, 'module'):  # DDP包装的模型
            self.model_device = self.model.module.device
        else:
            self.model_device = self.model.device
            
        print(f"[Rank {dist.get_rank() if dist.is_initialized() else 0}] Model device: {self.model_device}")
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 获取模型设备
        # print(f"compute_loss里的模型参数model_parameters是:{model.parameters()}")
        model_device = next(model.parameters()).device
        
        # 确保所有输入在模型设备上
        inputs_on_device = {}
        for key, value in inputs.items():
            if torch.is_tensor(value):
                # 使用 non_blocking=True 提高效率
                inputs_on_device[key] = value.to(model_device, non_blocking=True)
            else:
                inputs_on_device[key] = value
        
        import os
        import psutil

        current_pid = os.getpid()
        process = psutil.Process(current_pid)
        parent = process.parent()

        if parent:
            children = parent.children()
            if len(children) == 8:
                print("有8个训练进程在运行")
            else:
                print(f"有{len(children)}个训练进程在运行")
        else:
            print("无法确定父进程")
        # for key, value in inputs_on_device.items():
        #     if torch.is_tensor(value):
        #         print(f"  {key}: {value.device}")
        
        # 确保所有输入在模型设备上
        inputs_on_device = {
            "batch_pg": inputs["batch_pg"].to(model_device),
            "batch_ig": inputs["batch_ig"].to(model_device),
            "batch_rgb": inputs["batch_rgb"].to(model_device),
            "batch_depth": inputs["batch_depth"].to(model_device),
            "batch_labels": inputs["batch_labels"].to(model_device),
            "batch_augments": inputs["batch_augments"].to(model_device),
            "batch_label_critic": inputs["batch_label_critic"].to(model_device),
            "batch_augment_critic": inputs["batch_augment_critic"].to(model_device)
        }
        torch.cuda.synchronize(model_device)
        
        # 解包输入数据并移动到设备
        # batch_pg = inputs["batch_pg"]
        # batch_ig = inputs["batch_ig"]
        # batch_rgb = inputs["batch_rgb"]
        # batch_depth = inputs["batch_depth"]
        # batch_labels = inputs["batch_labels"]
        # batch_augments = inputs["batch_augments"]
        batch_label_critic = inputs["batch_label_critic"]
        batch_augment_critic = inputs["batch_augment_critic"]
        
        pred_ng, pred_pg, critic_pred, augment_pred, noise = model(
                inputs_on_device["batch_pg"],
                inputs_on_device["batch_ig"],
                inputs_on_device["batch_rgb"],
                inputs_on_device["batch_depth"],
                inputs_on_device["batch_labels"],
                inputs_on_device["batch_augments"]
            )
        
        ng_action_loss = (pred_ng - noise[0]).square().mean()
        pg_action_loss = (pred_pg - noise[1]).square().mean()
        # ig_action_loss = (pred_ig - noise[2]).square().mean()
        action_loss = 0.5 * pg_action_loss + 0.5 * ng_action_loss
        critic_loss = (critic_pred - batch_label_critic).square().mean() + \
                     (augment_pred - batch_augment_critic).square().mean()
        loss = 0.8 * action_loss + 0.2 * critic_loss
        
        outputs = {
            'pred_ng': pred_ng,
            'pred_pg': pred_pg,
            # 'pred_ig': pred_ig,
            'critic_pred': critic_pred,
            'augment_pred': augment_pred,
            'noise': noise,
            'loss': loss,
            'ng_action_loss': ng_action_loss,
            'pg_action_loss': pg_action_loss,
            # 'ig_action_loss': ig_action_loss,
            'critic_loss': critic_loss
        }
        # if self.logger:
        #     self.logger.info(
        #         f"[Step {self.state.global_step}] "
        #         f"Loss: {loss.item():.4f}, "
        #         f"Action Loss: {action_loss.item():.4f}, "
        #         f"Critic Loss: {critic_loss.item():.4f}"
        #     )

        
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        """创建并返回优化器"""
        rank = dist.get_rank() if dist.is_initialized() else 0
        
        # 获取学习率
        try:
            lr = self.config.il.lr
            if rank == 0:
                print(f"[Rank 0] Using learning rate: {lr}")
        except AttributeError:
            lr = 1e-4
            if rank == 0:
                print(f"[Rank 0] Warning: Using default learning rate: {lr}")
        
        # 确保模型在正确设备上
        if hasattr(self.model, 'module'):
            model_for_optim = self.model.module
        else:
            model_for_optim = self.model
            
        # 创建优化器
        optimizer = torch.optim.Adam(
            model_for_optim.parameters(), 
            lr=lr
        )
        
        if rank == 0:
            print(f"[Rank 0] Optimizer created with {len(optimizer.param_groups)} param groups")
            total_params = sum(p.numel() for p in model_for_optim.parameters() if p.requires_grad)
            print(f"[Rank 0] Total trainable parameters: {total_params:,}")
        


        return optimizer
    
    def create_scheduler(self, optimizer, num_training_steps: int):
        """创建学习率调度器"""
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.5,
            total_iters=10000
        )
        return scheduler
    
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """覆盖父类方法，完全控制创建过程"""
        print("\n=== 创建优化器和调度器 ===")
        
        # 创建优化器
        self.optimizer = self.create_optimizer()
        
        # 创建调度器（注意参数顺序）
        self.lr_scheduler = self.create_scheduler(self.optimizer, num_training_steps)
        
        return self.optimizer, self.lr_scheduler
    
    def get_train_dataloader(self):
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        sampler = DistributedSampler(self.train_dataset,
                                    num_replicas=world_size, 
                                    rank=rank,
                                    shuffle=True,
                                    seed=1234)
        
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.il.batch_size,
            sampler=sampler,
            num_workers=self.config.il.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.data_collator
        )
        # print(loader)
        return loader