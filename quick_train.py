#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速训练启动脚本 - 推荐配置
"""
import subprocess
import sys

# 推荐的训练配置（中等强度）
RECOMMENDED_CONFIG = {
    "load_model": "models/model_final.pth",
    "games": 30,
    "iterations": 200,
    "train_steps": 200,
    "batch_size": 64,
    "lr": 0.0005,
    "temperature": 0.7,
    "save_interval": 20,
    "save_dir": "models_v2"
}

def main():
    print("="*60)
    print("强化学习AI - 快速训练启动")
    print("="*60)
    print("\n当前配置（推荐 - 中等强度）:")
    for key, value in RECOMMENDED_CONFIG.items():
        print(f"  --{key.replace('_', '-')}: {value}")
    
    print("\n预计训练时间: 6-12小时")
    print("预期效果: 显著提升AI棋力")
    
    response = input("\n是否使用此配置开始训练? (y/n): ")
    
    if response.lower() != 'y':
        print("训练已取消")
        return
    
    # 构建命令
    cmd = [
        sys.executable,  # Python解释器路径
        "train_rl_ai.py",
    ]
    
    for key, value in RECOMMENDED_CONFIG.items():
        cmd.append(f"--{key.replace('_', '-')}")
        cmd.append(str(value))
    
    print("\n" + "="*60)
    print("开始训练...")
    print("="*60)
    print(f"命令: {' '.join(cmd)}\n")
    
    # 执行训练
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
