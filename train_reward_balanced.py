#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
奖励平衡修复脚本
解决黑方奖励异常高 (139.60) 的问题
"""
import subprocess
import sys

def run_training():
    """运行修复后的训练"""
    cmd = [
        sys.executable,
        'train_rl_ai_improved.py',
        '--games', '25',              
        '--iterations', '160',        
        '--batch-size', '32',         
        '--train-steps', '100',       
        '--save-interval', '32',      
        '--lr', '0.0005',             # 从0.0008降到0.0005
        '--weight-decay', '1e-4',     
        '--temperature', '0.9',       
        '--reward-scale', '0.01',     # 从0.08降到0.01（10倍削弱）
        '--num-channels', '64',       
        '--num-res-blocks', '6',      
        '--save-dir', 'models_reward_balanced'
    ]
    
    print("="*70)
    print("奖励平衡修复训练")
    print("="*70)
    print("\n🔧 关键改动:")
    print(f"  奖励缩放: 0.08 → 0.01 (10倍削弱)")
    print(f"  学习率: 0.0008 → 0.0005")
    print(f"  权重衰减: 2e-5 → 1e-4")
    print(f"\n📝 奖励幅度调整:")
    print(f"  吃车: 10.8 → 1.35")
    print(f"  解将: 8.0 → 1.0")
    print(f"  翻子: 0.4 → 0.05")
    print(f"  走入将: -4.0 → -0.5")
    print(f"\n📊 预期效果:")
    print(f"  原: (胜者: black, 红方: 4.80, 黑方: 139.60) ❌")
    print(f"  修: (胜者: black, 红方: -2.00, 黑方: 3.50) ✓")
    print(f"\n✓ 修复点:")
    print(f"  - 输家奖励应该为负")
    print(f"  - 累积奖励在 ±2～±15 范围内")
    print(f"  - 赢家和输家奖励符号相反")
    print("="*70 + "\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "="*70)
        print("✓ 训练完成！")
        print("="*70)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\n✗ 训练失败: {e}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n⚠ 训练被中断")
        return 1

if __name__ == "__main__":
    sys.exit(run_training())
