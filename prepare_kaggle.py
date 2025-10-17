#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
准备 Kaggle 上传包
将项目打包成可在 Kaggle 上运行的格式
"""
import os
import zipfile
import shutil
from datetime import datetime

def create_kaggle_package():
    """创建 Kaggle 上传包"""
    
    print("="*70)
    print("准备 Kaggle 上传包")
    print("="*70)
    
    # 需要包含的文件
    include_files = {
        # 训练脚本
        'train_rl_ai_optimized.py': '优化训练脚本',
        'kaggle_setup.py': 'Kaggle 环境配置',
        
        # AI 模块
        'ai/__init__.py': 'AI 模块初始化',
        'ai/ai_player.py': 'AI 玩家基类',
        'ai/mcts_ai.py': 'MCTS AI',
        
        # 核心模块
        'core/__init__.py': '核心模块初始化',
        'core/game_state.py': '游戏状态',
        'core/interfaces.py': '接口定义',
        
        # 游戏逻辑
        'game/__init__.py': '游戏模块初始化',
        'game/dark_chess_board.py': '棋盘',
        'game/dark_chess_piece.py': '棋子',
        'game/game_engine.py': '游戏引擎',
        'game/zobrist_hash.py': 'Zobrist 哈希',
        
        # 玩家
        'players/__init__.py': '玩家模块初始化',
        'players/ai_player.py': 'AI 玩家',
        'players/base_player.py': '玩家基类',
        'players/human_player.py': '人类玩家',
        
        # 强化学习
        'rl_ai/__init__.py': 'RL 模块初始化',
        'rl_ai/neural_network.py': '神经网络',
        'rl_ai/rl_player.py': 'RL 玩家',
        'rl_ai/rl_trainer.py': 'RL 训练器',
        
        # 文档
        'README.md': '项目说明',
        'QUICK_START_OPTIMIZED.md': '快速开始',
        'KAGGLE_TRAINING_GUIDE.md': 'Kaggle 训练指南',
    }
    
    # 输出文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_zip = f'xq_kaggle_{timestamp}.zip'
    
    # 统计
    total_files = 0
    missing_files = []
    total_size = 0
    
    print(f"\n正在打包到: {output_zip}\n")
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path, description in include_files.items():
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                total_size += file_size
                zipf.write(file_path)
                print(f"  ✓ {file_path:<50} ({file_size:>8,} bytes) - {description}")
                total_files += 1
            else:
                print(f"  ✗ {file_path:<50} (不存在)")
                missing_files.append(file_path)
    
    print("\n" + "="*70)
    print(f"打包完成!")
    print("="*70)
    print(f"输出文件: {output_zip}")
    print(f"文件数量: {total_files}")
    print(f"总大小: {total_size / 1024:.1f} KB ({total_size:,} bytes)")
    print(f"压缩后: {os.path.getsize(output_zip) / 1024:.1f} KB")
    
    if missing_files:
        print(f"\n⚠ 缺失文件: {len(missing_files)}")
        for f in missing_files:
            print(f"  - {f}")
    
    print("\n" + "="*70)
    print("下一步:")
    print("="*70)
    print("1. 访问 https://www.kaggle.com/datasets")
    print("2. 点击 'New Dataset'")
    print(f"3. 上传 {output_zip}")
    print("4. 填写信息:")
    print("   - Title: Chinese Dark Chess RL Training")
    print("   - Description: 揭棋强化学习 AI 训练代码")
    print("5. 点击 'Create'")
    print("\n在 Notebook 中使用:")
    print("="*70)
    print("# 解压项目")
    print("!unzip -q /kaggle/input/your-dataset-name/*.zip -d /kaggle/working/")
    print("!cd /kaggle/working && python kaggle_setup.py")
    print("\n# 开始训练")
    print("!cd /kaggle/working && python train_rl_ai_optimized.py \\")
    print("    --games 100 --iterations 500 --batch-size 512 \\")
    print("    --parallel-games 8 --save-dir /kaggle/working/models")
    print("="*70)
    
    return output_zip

def create_kaggle_notebook():
    """创建 Kaggle Notebook 模板"""
    
    notebook_content = """# 揭棋 AI 训练 - Kaggle Notebook

## 1. 环境配置

```python
# 解压项目文件
!unzip -q /kaggle/input/chinese-dark-chess-rl-training/*.zip -d /kaggle/working/
!ls /kaggle/working/

# 配置环境
!cd /kaggle/working && python kaggle_setup.py
```

## 2. 检查 GPU

```python
!nvidia-smi
```

## 3. 开始训练 (快速测试)

```python
# 小规模测试配置
!cd /kaggle/working && python train_rl_ai_optimized.py \\
    --games 10 \\
    --iterations 5 \\
    --train-steps 50 \\
    --batch-size 256 \\
    --parallel-games 4 \\
    --num-channels 128 \\
    --num-res-blocks 10 \\
    --save-dir /kaggle/working/models_test
```

## 4. 正式训练 (高性能配置)

```python
# 这将运行较长时间，建议在确认环境正常后再运行
!cd /kaggle/working && python train_rl_ai_optimized.py \\
    --games 100 \\
    --iterations 500 \\
    --train-steps 500 \\
    --batch-size 512 \\
    --parallel-games 8 \\
    --num-channels 256 \\
    --num-res-blocks 20 \\
    --save-dir /kaggle/working/models \\
    --save-interval 20
```

## 5. 监控训练进度

```python
import time
import os
from IPython.display import clear_output

# 持续监控
while True:
    clear_output(wait=True)
    
    # GPU 状态
    print("GPU 状态:")
    !nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv
    
    # 模型文件
    print("\\n已保存的模型:")
    !ls -lh /kaggle/working/models/*.pth 2>/dev/null || echo "暂无模型"
    
    # 等待
    time.sleep(30)
```

## 6. 下载模型

```python
from IPython.display import FileLink
import os

# 打包所有模型
!cd /kaggle/working && tar -czf models.tar.gz models/

# 提供下载链接
display(FileLink('/kaggle/working/models.tar.gz'))

# 或者下载单个模型
for f in os.listdir('/kaggle/working/models'):
    if f.endswith('.pth'):
        display(FileLink(f'/kaggle/working/models/{f}'))
```

## 7. 继续训练 (断点续训)

```python
# 如果之前训练中断，可以加载最新的检查点继续训练
!cd /kaggle/working && python train_rl_ai_optimized.py \\
    --load-model /kaggle/working/models/model_iter_100.pth \\
    --games 100 \\
    --iterations 400 \\
    --batch-size 512 \\
    --parallel-games 8 \\
    --save-dir /kaggle/working/models
```

## 提示

- Kaggle Notebook 最长运行 12 小时
- 每 20 轮会自动保存模型
- 建议定期下载模型到本地备份
- GPU T4: 适合批次 256-512
- GPU P100: 适合批次 512-1024
"""
    
    notebook_file = 'kaggle_notebook_template.md'
    with open(notebook_file, 'w', encoding='utf-8') as f:
        f.write(notebook_content)
    
    print(f"\n✓ 创建 Notebook 模板: {notebook_file}")
    return notebook_file

def main():
    # 创建上传包
    zip_file = create_kaggle_package()
    
    # 创建 Notebook 模板
    notebook_file = create_kaggle_notebook()
    
    print(f"\n{'='*70}")
    print("准备完成！")
    print(f"{'='*70}")
    print(f"上传包: {zip_file}")
    print(f"Notebook 模板: {notebook_file}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
