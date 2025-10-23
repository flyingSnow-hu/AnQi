#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
价值损失修复训练脚本
针对价值损失异常高的问题优化
"""
import subprocess
import sys

def run_training():
    """运行优化后的训练"""
    cmd = [
        sys.executable,
        'train_rl_ai_improved.py',
        '--games', '25',              
        '--iterations', '160',        
        '--batch-size', '16',         # 降低batch_size，梯度更稳定
        '--train-steps', '100',       
        '--save-interval', '32',      
        '--lr', '0.0003',             # ⬇️ 关键：从0.0008降到0.0003（3.7倍）
        '--weight-decay', '1e-4',     # ⬆️ 增加正则化
        '--temperature', '0.9',       
        '--reward-scale', '0.01',     # ⬇️ 从0.08降到0.01（降低奖励幅度）
        '--num-channels', '64',       
        '--num-res-blocks', '6',      
        '--save-dir', 'models_value_loss_fix'
    ]
    
    print("="*70)
    print("价值损失修复训练 - 参数优化版")
    print("="*70)
    print("\n🔧 关键改动:")
    print(f"  学习率: 0.0008 → 0.0003 (降低3.7倍)")
    print(f"  权重衰减: 2e-5 → 1e-4 (增加5倍)")
    print(f"  batch_size: 32 → 16 (更稳定的梯度)")
    print(f"  奖励缩放: 0.08 → 0.01 (降低奖励幅度)")
    print(f"\n📊 代码修改:")
    print(f"  价值计算: final_value + step_reward")
    print(f"         → final_value×0.7 + step_reward×0.03")
    print(f"  (防止单步奖励过大导致价值爆炸)")
    print(f"\n📈 预期效果:")
    print(f"  策略损失: 0.50 (保持)")
    print(f"  价值损失: 69.48 → 0.3-0.5 ✓ (大幅改善)")
    print(f"  总损失: 69.98 → 0.8-1.0 ✓")
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
