#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能测试 - 对比不同配置的训练速度
"""
import sys
import os
import time
import torch

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from rl_ai.neural_network import DarkChessNet, BoardEncoder
import numpy as np


def test_inference_speed(batch_size, num_channels, num_res_blocks, num_batches=100):
    """测试推理速度"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型
    model = DarkChessNet(num_channels=num_channels, num_res_blocks=num_res_blocks)
    model.to(device)
    model.eval()
    
    # 生成随机数据
    encoder = BoardEncoder()
    dummy_input = torch.randn(batch_size, 18, 10, 9).to(device)
    
    # 预热
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # 计时
    start_time = time.time()
    for _ in range(num_batches):
        with torch.no_grad():
            _ = model(dummy_input)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    elapsed = time.time() - start_time
    samples_per_sec = (batch_size * num_batches) / elapsed
    
    return samples_per_sec, elapsed


def test_training_speed(batch_size, num_channels, num_res_blocks, num_steps=100):
    """测试训练速度"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型和优化器
    model = DarkChessNet(num_channels=num_channels, num_res_blocks=num_res_blocks)
    model.to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 生成随机数据
    dummy_input = torch.randn(batch_size, 18, 10, 9).to(device)
    dummy_policy = torch.randn(batch_size, 8100).to(device)
    dummy_value = torch.randn(batch_size, 1).to(device)
    
    # 预热
    for _ in range(10):
        optimizer.zero_grad()
        policy_logits, values = model(dummy_input)
        loss = torch.nn.functional.mse_loss(values, dummy_value)
        loss.backward()
        optimizer.step()
    
    # 计时
    start_time = time.time()
    for _ in range(num_steps):
        optimizer.zero_grad()
        policy_logits, values = model(dummy_input)
        
        # 计算损失
        policy_loss = -torch.mean(torch.sum(
            dummy_policy * torch.nn.functional.log_softmax(policy_logits, dim=1), 
            dim=1))
        value_loss = torch.nn.functional.mse_loss(values, dummy_value)
        total_loss = policy_loss + value_loss
        
        total_loss.backward()
        optimizer.step()
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    elapsed = time.time() - start_time
    steps_per_sec = num_steps / elapsed
    samples_per_sec = (batch_size * num_steps) / elapsed
    
    return steps_per_sec, samples_per_sec, elapsed


def test_mixed_precision_speedup(batch_size, num_channels, num_res_blocks, num_steps=100):
    """测试混合精度训练加速"""
    from torch.cuda.amp import autocast, GradScaler
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != 'cuda':
        print("混合精度需要CUDA")
        return None, None
    
    model = DarkChessNet(num_channels=num_channels, num_res_blocks=num_res_blocks)
    model.to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler()
    
    dummy_input = torch.randn(batch_size, 18, 10, 9).to(device)
    dummy_policy = torch.randn(batch_size, 8100).to(device)
    dummy_value = torch.randn(batch_size, 1).to(device)
    
    # 预热
    for _ in range(10):
        optimizer.zero_grad()
        with autocast():
            policy_logits, values = model(dummy_input)
            loss = torch.nn.functional.mse_loss(values, dummy_value)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    # 计时
    start_time = time.time()
    for _ in range(num_steps):
        optimizer.zero_grad()
        with autocast():
            policy_logits, values = model(dummy_input)
            policy_loss = -torch.mean(torch.sum(
                dummy_policy * torch.nn.functional.log_softmax(policy_logits, dim=1), 
                dim=1))
            value_loss = torch.nn.functional.mse_loss(values, dummy_value)
            total_loss = policy_loss + value_loss
        
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    torch.cuda.synchronize()
    
    elapsed = time.time() - start_time
    steps_per_sec = num_steps / elapsed
    
    return steps_per_sec, elapsed


def get_model_size(num_channels, num_res_blocks):
    """获取模型参数量"""
    model = DarkChessNet(num_channels=num_channels, num_res_blocks=num_res_blocks)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def main():
    print("="*70)
    print("GPU性能测试 - 优化配置对比")
    print("="*70)
    
    # 显示设备信息
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("\n警告: 未检测到CUDA，使用CPU模式")
    
    print(f"PyTorch版本: {torch.__version__}")
    
    # 测试配置
    configs = [
        {"name": "基础配置", "batch": 128, "channels": 128, "blocks": 10},
        {"name": "中等配置", "batch": 256, "channels": 256, "blocks": 20},
        {"name": "高级配置", "batch": 512, "channels": 256, "blocks": 20},
        {"name": "极限配置", "batch": 512, "channels": 384, "blocks": 30},
    ]
    
    print("\n" + "="*70)
    print("配置信息")
    print("="*70)
    
    for config in configs:
        total, trainable = get_model_size(config['channels'], config['blocks'])
        print(f"\n{config['name']}:")
        print(f"  批次大小: {config['batch']}")
        print(f"  网络: {config['channels']}通道 x {config['blocks']}残差块")
        print(f"  参数量: {total/1e6:.1f}M ({trainable/1e6:.1f}M可训练)")
    
    # 推理速度测试
    print("\n" + "="*70)
    print("推理速度测试")
    print("="*70)
    print(f"{'配置':<15} {'批次':<10} {'样本/秒':<15} {'相对速度':<10}")
    print("-"*70)
    
    baseline_inference = None
    for config in configs:
        try:
            samples_per_sec, elapsed = test_inference_speed(
                config['batch'], 
                config['channels'], 
                config['blocks'],
                num_batches=50
            )
            
            if baseline_inference is None:
                baseline_inference = samples_per_sec
                relative = 1.0
            else:
                relative = samples_per_sec / baseline_inference
            
            print(f"{config['name']:<15} {config['batch']:<10} {samples_per_sec:<15.0f} {relative:<10.2f}x")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"{config['name']:<15} {config['batch']:<10} {'显存不足':<15} {'-':<10}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                raise
    
    # 训练速度测试
    print("\n" + "="*70)
    print("训练速度测试 (FP32)")
    print("="*70)
    print(f"{'配置':<15} {'批次':<10} {'步数/秒':<12} {'样本/秒':<15} {'相对速度':<10}")
    print("-"*70)
    
    baseline_training = None
    for config in configs:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            steps_per_sec, samples_per_sec, elapsed = test_training_speed(
                config['batch'], 
                config['channels'], 
                config['blocks'],
                num_steps=50
            )
            
            if baseline_training is None:
                baseline_training = samples_per_sec
                relative = 1.0
            else:
                relative = samples_per_sec / baseline_training
            
            print(f"{config['name']:<15} {config['batch']:<10} {steps_per_sec:<12.1f} "
                  f"{samples_per_sec:<15.0f} {relative:<10.2f}x")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"{config['name']:<15} {config['batch']:<10} {'显存不足':<12} {'-':<15} {'-':<10}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                raise
    
    # 混合精度加速测试
    if torch.cuda.is_available():
        print("\n" + "="*70)
        print("混合精度训练 (FP16) 加速测试")
        print("="*70)
        print(f"{'配置':<15} {'FP32步/秒':<15} {'FP16步/秒':<15} {'加速比':<10}")
        print("-"*70)
        
        # 测试中等配置
        config = configs[1]  # 中等配置
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        try:
            # FP32
            steps_fp32, _, _ = test_training_speed(
                config['batch'], config['channels'], config['blocks'], num_steps=50
            )
            
            torch.cuda.empty_cache()
            
            # FP16
            steps_fp16, _ = test_mixed_precision_speedup(
                config['batch'], config['channels'], config['blocks'], num_steps=50
            )
            
            speedup = steps_fp16 / steps_fp32
            print(f"{config['name']:<15} {steps_fp32:<15.1f} {steps_fp16:<15.1f} {speedup:<10.2f}x")
            
        except Exception as e:
            print(f"混合精度测试失败: {e}")
    
    # 显存使用统计
    if torch.cuda.is_available():
        print("\n" + "="*70)
        print("显存使用统计")
        print("="*70)
        
        for config in configs:
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                # 创建模型并进行一次前向传播
                model = DarkChessNet(config['channels'], config['blocks'])
                model.to('cuda')
                dummy_input = torch.randn(config['batch'], 18, 10, 9).to('cuda')
                
                with torch.no_grad():
                    _ = model(dummy_input)
                
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                peak = torch.cuda.max_memory_allocated() / 1024**3
                
                print(f"\n{config['name']}:")
                print(f"  分配: {allocated:.2f} GB")
                print(f"  保留: {reserved:.2f} GB")
                print(f"  峰值: {peak:.2f} GB")
                
                del model, dummy_input
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n{config['name']}: 显存不足")
                torch.cuda.empty_cache()
    
    print("\n" + "="*70)
    print("测试完成!")
    print("="*70)
    
    # 推荐配置
    print("\n基于测试结果的推荐:")
    if torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n您的GPU有 {total_mem:.1f} GB 显存")
        
        if total_mem >= 10:
            print("推荐使用: 高级配置或极限配置")
            print("  --batch-size 512 --num-channels 256-384 --num-res-blocks 20-30")
        elif total_mem >= 6:
            print("推荐使用: 中等配置")
            print("  --batch-size 256-384 --num-channels 256 --num-res-blocks 20")
        else:
            print("推荐使用: 基础配置")
            print("  --batch-size 128-256 --num-channels 128-256 --num-res-blocks 10-15")


if __name__ == "__main__":
    main()
