# -*- coding: utf-8 -*-
"""
早停机制实现
"""
import torch
import numpy as np
import os


class EarlyStopping:
    """早停机制，当验证集性能不再提升时停止训练"""
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True, 
                 mode='max', save_best_model=True, model_path=None):
        """
        Args:
            patience (int): 等待多少个epoch没有改善就停止训练
            min_delta (float): 最小改善阈值
            restore_best_weights (bool): 是否恢复最佳权重
            mode (str): 'min' 或 'max'，指标是越小越好还是越大越好
            save_best_model (bool): 是否保存最佳模型
            model_path (str): 模型保存路径
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        self.save_best_model = save_best_model
        self.model_path = model_path
        
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        
        if mode == 'min':
            self.monitor_op = np.less
            self.best_score = np.Inf
        else:
            self.monitor_op = np.greater
            self.best_score = -np.Inf
    
    def __call__(self, current_score, model, epoch):
        """
        检查是否需要早停
        
        Args:
            current_score: 当前验证指标
            model: 当前模型
            epoch: 当前epoch
            
        Returns:
            bool: 是否需要停止训练
        """
        if self.monitor_op(current_score - self.min_delta, self.best_score):
            self.best_score = current_score
            self.wait = 0
            
            # 保存最佳权重
            if self.restore_best_weights:
                self.best_weights = {
                    name: param.clone() for name, param in model.named_parameters()
                }
            
            # 保存最佳模型到文件（只在主进程中保存）
            if self.save_best_model and self.model_path:
                # 检查是否在分布式环境中，如果是则只在主进程保存
                try:
                    import util.misc as utils
                    if not hasattr(utils, 'is_main_process') or utils.is_main_process():
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'epoch': epoch,
                            'best_score': self.best_score
                        }, self.model_path)
                except:
                    # 如果无法导入utils或检查主进程，直接保存（向后兼容）
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'epoch': epoch,
                        'best_score': self.best_score
                    }, self.model_path)
                
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                return True
        
        return False
    
    def restore_weights(self, model):
        """恢复最佳权重"""
        if self.best_weights is not None:
            for name, param in model.named_parameters():
                param.data.copy_(self.best_weights[name])


class TwoStageTrainer:
    """两阶段训练器"""
    
    def __init__(self, model, criterion, optimizer, lr_scheduler, 
                 stage1_epochs=50, stage2_epochs=100,
                 stage1_patience=10, stage2_patience=15,
                 output_dir="./checkpoints"):
        """
        Args:
            model: 模型
            criterion: 损失函数
            optimizer: 优化器
            lr_scheduler: 学习率调度器
            stage1_epochs: 第一阶段最大训练轮数
            stage2_epochs: 第二阶段最大训练轮数
            stage1_patience: 第一阶段早停耐心值
            stage2_patience: 第二阶段早停耐心值
            output_dir: 模型保存目录
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        
        self.stage1_epochs = stage1_epochs
        self.stage2_epochs = stage2_epochs
        
        # 创建保存目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 第一阶段早停
        self.stage1_early_stopping = EarlyStopping(
            patience=stage1_patience,
            mode='max',  # 假设AP越大越好
            save_best_model=True,
            model_path=os.path.join(output_dir, "stage1_best_model.pth")
        )
        
        # 第二阶段早停
        self.stage2_early_stopping = EarlyStopping(
            patience=stage2_patience,
            mode='max',
            save_best_model=True,
            model_path=os.path.join(output_dir, "stage2_best_model.pth")
        )
        
        self.current_stage = 1
        self.stage1_completed = False
        
    def should_stop_stage1(self, val_score, epoch):
        """检查第一阶段是否应该停止"""
        return self.stage1_early_stopping(val_score, self.model, epoch)
    
    def should_stop_stage2(self, val_score, epoch):
        """检查第二阶段是否应该停止"""
        return self.stage2_early_stopping(val_score, self.model, epoch)
    
    def complete_stage1(self):
        """完成第一阶段训练"""
        self.stage1_completed = True
        self.current_stage = 2
        
        # 从文件加载第一阶段的最佳权重
        if self.stage1_early_stopping.model_path and os.path.exists(self.stage1_early_stopping.model_path):
            print(f"加载第一阶段最佳模型权重: {self.stage1_early_stopping.model_path}")
            checkpoint = torch.load(self.stage1_early_stopping.model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"第一阶段训练完成，最佳验证AP: {checkpoint.get('best_score', 'N/A')}")
        else:
            # 如果文件不存在，尝试从内存恢复
            print("警告: 第一阶段最佳模型文件不存在，尝试从内存恢复权重")
            self.stage1_early_stopping.restore_weights(self.model)
            print(f"第一阶段训练完成，最佳验证AP: {self.stage1_early_stopping.best_score:.4f}")
        
        print("开始第二阶段训练...")
    
    def get_stage_info(self):
        """获取当前阶段信息"""
        return {
            'current_stage': self.current_stage,
            'stage1_completed': self.stage1_completed,
            'stage1_best_score': self.stage1_early_stopping.best_score,
            'stage2_best_score': self.stage2_early_stopping.best_score if self.current_stage == 2 else None
        }

