#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU性能监控工具
显示GPU利用率、显存使用、温度等信息
"""
import sys
import os
import time

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    import torch
    import pynvml
    has_nvml = True
    try:
        pynvml.nvmlInit()
    except:
        has_nvml = False
        print("警告: 无法初始化NVML，部分GPU信息可能不可用")
except ImportError:
    print("错误: 需要安装 nvidia-ml-py3")
    print("运行: pip install nvidia-ml-py3")
    sys.exit(1)


def get_gpu_info():
    """获取GPU信息"""
    if not torch.cuda.is_available():
        return None
    
    info = {
        'name': torch.cuda.get_device_name(0),
        'cuda_version': torch.version.cuda,
        'pytorch_version': torch.__version__,
        'device_count': torch.cuda.device_count()
    }
    
    # 显存信息
    memory_allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
    memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
    memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    info['memory_allocated'] = memory_allocated
    info['memory_reserved'] = memory_reserved
    info['memory_total'] = memory_total
    info['memory_free'] = memory_total - memory_reserved
    
    # NVML额外信息
    if has_nvml:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # GPU利用率
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            info['gpu_util'] = util.gpu
            info['mem_util'] = util.memory
            
            # 温度
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            info['temperature'] = temp
            
            # 功率
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # W
            power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
            info['power'] = power
            info['power_limit'] = power_limit
            
            # 风扇转速
            try:
                fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
                info['fan_speed'] = fan_speed
            except:
                info['fan_speed'] = None
                
        except Exception as e:
            print(f"获取NVML信息失败: {e}")
    
    return info


def print_gpu_info(info, clear_screen=True):
    """打印GPU信息"""
    if clear_screen:
        os.system('cls' if os.name == 'nt' else 'clear')
    
    print("="*70)
    print(" GPU 性能监控")
    print("="*70)
    print(f"GPU型号: {info['name']}")
    print(f"CUDA版本: {info['cuda_version']}")
    print(f"PyTorch版本: {info['pytorch_version']}")
    print(f"GPU数量: {info['device_count']}")
    print("-"*70)
    
    # 显存使用
    mem_percent = (info['memory_reserved'] / info['memory_total']) * 100
    print(f"显存使用: {info['memory_reserved']:.2f} GB / {info['memory_total']:.2f} GB ({mem_percent:.1f}%)")
    print(f"  - 已分配: {info['memory_allocated']:.2f} GB")
    print(f"  - 空闲: {info['memory_free']:.2f} GB")
    
    # GPU利用率
    if 'gpu_util' in info:
        print(f"\nGPU利用率: {info['gpu_util']}%")
        print(f"显存控制器利用率: {info['mem_util']}%")
    
    # 温度
    if 'temperature' in info:
        temp_bar = "█" * int(info['temperature'] / 5) + "░" * (20 - int(info['temperature'] / 5))
        print(f"\n温度: {info['temperature']}°C [{temp_bar}]")
    
    # 功率
    if 'power' in info:
        power_percent = (info['power'] / info['power_limit']) * 100
        power_bar = "█" * int(power_percent / 5) + "░" * (20 - int(power_percent / 5))
        print(f"功率: {info['power']:.1f}W / {info['power_limit']:.1f}W ({power_percent:.1f}%)")
        print(f"      [{power_bar}]")
    
    # 风扇
    if info.get('fan_speed') is not None:
        fan_bar = "█" * int(info['fan_speed'] / 5) + "░" * (20 - int(info['fan_speed'] / 5))
        print(f"\n风扇转速: {info['fan_speed']}%")
        print(f"         [{fan_bar}]")
    
    print("="*70)
    
    # 优化建议
    if info.get('gpu_util', 100) < 70:
        print("\n⚠ GPU利用率较低，建议:")
        print("  - 增加 --parallel-games (并行游戏数)")
        print("  - 增加 --batch-size (批次大小)")
        print("  - 增加 --num-channels 或 --num-res-blocks (网络规模)")
    
    if mem_percent < 50:
        print("\n💡 显存利用率较低，可以:")
        print("  - 增加批次大小到 512 或更高")
        print("  - 增加网络通道数到 512")
        print("  - 增加残差块数量到 30+")


def monitor_loop(interval=2):
    """持续监控GPU"""
    print("开始监控GPU (按Ctrl+C退出)...\n")
    
    try:
        while True:
            info = get_gpu_info()
            if info:
                print_gpu_info(info, clear_screen=True)
            else:
                print("未检测到GPU")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n监控已停止")
        if has_nvml:
            pynvml.nvmlShutdown()


def show_quick_stats():
    """显示快速统计"""
    info = get_gpu_info()
    if not info:
        print("未检测到GPU")
        return
    
    print_gpu_info(info, clear_screen=False)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='GPU性能监控')
    parser.add_argument('--interval', type=int, default=2, help='监控刷新间隔(秒)')
    parser.add_argument('--once', action='store_true', help='只显示一次，不持续监控')
    
    args = parser.parse_args()
    
    if args.once:
        show_quick_stats()
    else:
        monitor_loop(args.interval)
