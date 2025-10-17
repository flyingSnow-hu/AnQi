#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kaggle 环境自动配置脚本
"""
import os
import sys
import subprocess
import platform

def is_kaggle():
    """检测是否在 Kaggle 环境"""
    return os.path.exists('/kaggle')

def is_colab():
    """检测是否在 Google Colab 环境"""
    try:
        import google.colab
        return True
    except:
        return False

def get_gpu_info():
    """获取 GPU 信息"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', 
                                '--format=csv,noheader'], 
                               capture_output=True, text=True)
        return result.stdout.strip()
    except:
        return "未检测到 GPU"

def setup_kaggle():
    """配置 Kaggle 环境"""
    print("="*70)
    print("Kaggle 环境配置")
    print("="*70)
    
    # GPU 信息
    print(f"\nGPU 信息:")
    print(get_gpu_info())
    
    # Python 版本
    print(f"\nPython 版本: {sys.version}")
    
    # 检查 PyTorch
    try:
        import torch
        print(f"\nPyTorch 版本: {torch.__version__}")
        print(f"CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA 版本: {torch.version.cuda}")
            print(f"设备数量: {torch.cuda.device_count()}")
    except ImportError:
        print("\n⚠ PyTorch 未安装，正在安装...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision'])
    
    # 创建工作目录
    work_dirs = [
        '/kaggle/working/models',
        '/kaggle/working/logs'
    ]
    
    for dir_path in work_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ 创建目录: {dir_path}")
    
    print("\n" + "="*70)
    print("✓ Kaggle 环境配置完成！")
    print("="*70)
    
    # 推荐配置
    print("\n推荐训练命令:")
    print("-"*70)
    print("python train_rl_ai_optimized.py \\")
    print("    --games 100 \\")
    print("    --iterations 500 \\")
    print("    --batch-size 512 \\")
    print("    --parallel-games 8 \\")
    print("    --num-channels 256 \\")
    print("    --num-res-blocks 20 \\")
    print("    --save-dir /kaggle/working/models \\")
    print("    --save-interval 20")
    print("-"*70)

def setup_colab():
    """配置 Google Colab 环境"""
    print("="*70)
    print("Google Colab 环境配置")
    print("="*70)
    
    # GPU 信息
    print(f"\nGPU 信息:")
    print(get_gpu_info())
    
    # 挂载 Google Drive
    try:
        from google.colab import drive
        drive_path = '/content/drive'
        if not os.path.exists(f'{drive_path}/MyDrive'):
            print("\n正在挂载 Google Drive...")
            drive.mount(drive_path)
        
        # 创建工作目录
        work_dir = f'{drive_path}/MyDrive/xq_training'
        os.makedirs(work_dir, exist_ok=True)
        print(f"✓ 工作目录: {work_dir}")
        
        print("\n推荐训练命令:")
        print("-"*70)
        print("python train_rl_ai_optimized.py \\")
        print(f"    --save-dir {work_dir}/models \\")
        print("    --games 80 \\")
        print("    --batch-size 384 \\")
        print("    --parallel-games 6")
        print("-"*70)
        
    except ImportError:
        print("\n⚠ 不在 Colab 环境中")

def setup_local():
    """本地环境配置"""
    print("="*70)
    print("本地环境")
    print("="*70)
    
    print(f"\n操作系统: {platform.system()} {platform.release()}")
    print(f"Python 版本: {sys.version}")
    
    # GPU 信息
    print(f"\nGPU 信息:")
    print(get_gpu_info())
    
    # 检查虚拟环境
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print(f"\n✓ 虚拟环境: {sys.prefix}")
    else:
        print(f"\n⚠ 未使用虚拟环境")
    
    print("\n✓ 本地环境已就绪")

def main():
    print("\n环境检测中...")
    
    if is_kaggle():
        setup_kaggle()
    elif is_colab():
        setup_colab()
    else:
        setup_local()
    
    print("\n")

if __name__ == "__main__":
    main()
