"""
强化学习训练器
通过自对弈生成训练数据，使用策略梯度更新网络
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import deque
import os
import json
from datetime import datetime

from .neural_network import DarkChessNet, BoardEncoder


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state, policy, value, winner):
        """
        添加一条训练样本
        state: 棋盘状态编码 (18, 10, 9)
        policy: MCTS改进后的策略 (8100,)
        value: 最终胜负结果 (1/-1/0)
        winner: 获胜方 ("red"/"black"/None)
        """
        self.buffer.append((state, policy, value, winner))
    
    def sample(self, batch_size):
        """随机采样一批数据"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states = np.array([x[0] for x in batch])
        policies = np.array([x[1] for x in batch])
        values = np.array([x[2] for x in batch])
        
        return states, policies, values
    
    def __len__(self):
        return len(self.buffer)


class RLTrainer:
    """强化学习训练器 - AlphaZero风格"""
    
    def __init__(self, model=None, lr=0.001, weight_decay=1e-4):
        """
        初始化训练器
        model: DarkChessNet模型，如果为None则创建新模型
        lr: 学习率
        weight_decay: 权重衰减
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        if model is None:
            self.model = DarkChessNet(num_channels=128, num_res_blocks=10)
        else:
            self.model = model
        
        self.model.to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # 经验回放
        self.replay_buffer = ReplayBuffer(max_size=100000)
        
        # 训练统计
        self.training_stats = {
            'episodes': 0,
            'total_games': 0,
            'policy_losses': [],
            'value_losses': [],
            'total_losses': []
        }
    
    def train_step(self, batch_size=64):
        """
        执行一步训练
        从replay buffer中采样数据，更新网络参数
        """
        if len(self.replay_buffer) < batch_size:
            return None
        
        self.model.train()
        
        # 采样数据
        states, target_policies, target_values = self.replay_buffer.sample(batch_size)
        
        # 转换为tensor
        states = torch.FloatTensor(states).to(self.device)
        target_policies = torch.FloatTensor(target_policies).to(self.device)
        target_values = torch.FloatTensor(target_values).unsqueeze(1).to(self.device)
        
        # 前向传播
        policy_logits, values = self.model(states)
        
        # 计算损失
        # 策略损失：交叉熵
        policy_loss = -torch.mean(torch.sum(target_policies * F.log_softmax(policy_logits, dim=1), dim=1))
        
        # 价值损失：均方误差
        value_loss = F.mse_loss(values, target_values)
        
        # 总损失
        total_loss = policy_loss + value_loss
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # 记录统计
        self.training_stats['policy_losses'].append(policy_loss.item())
        self.training_stats['value_losses'].append(value_loss.item())
        self.training_stats['total_losses'].append(total_loss.item())
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def train_epoch(self, num_steps=100, batch_size=64):
        """训练一个epoch"""
        losses = []
        for _ in range(num_steps):
            loss = self.train_step(batch_size)
            if loss:
                losses.append(loss)
        
        if losses:
            avg_loss = {
                'policy_loss': np.mean([l['policy_loss'] for l in losses]),
                'value_loss': np.mean([l['value_loss'] for l in losses]),
                'total_loss': np.mean([l['total_loss'] for l in losses])
            }
            return avg_loss
        return None
    
    def save_checkpoint(self, filepath, epoch=0):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats
        }
        torch.save(checkpoint, filepath)
        print(f"模型已保存到: {filepath}")
    
    def load_checkpoint(self, filepath):
        """加载模型检查点"""
        if not os.path.exists(filepath):
            print(f"文件不存在: {filepath}")
            return False
        
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint.get('training_stats', self.training_stats)
        
        print(f"模型已从 {filepath} 加载")
        return True
    
    def get_stats(self):
        """获取训练统计"""
        if not self.training_stats['total_losses']:
            return None
        
        return {
            'episodes': self.training_stats['episodes'],
            'total_games': self.training_stats['total_games'],
            'avg_policy_loss': np.mean(self.training_stats['policy_losses'][-100:]),
            'avg_value_loss': np.mean(self.training_stats['value_losses'][-100:]),
            'avg_total_loss': np.mean(self.training_stats['total_losses'][-100:]),
            'buffer_size': len(self.replay_buffer)
        }


import torch.nn.functional as F
