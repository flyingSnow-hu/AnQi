# Kaggle 云端训练指南

## 🌐 为什么使用 Kaggle

Kaggle 提供免费的 GPU 资源用于训练深度学习模型：
- ✅ **免费 GPU**: Tesla P100 或 T4 (16GB 显存)
- ✅ **每周 30 小时**: GPU 使用时长
- ✅ **持久化存储**: 可以保存模型
- ✅ **Jupyter 环境**: 方便调试
- ✅ **无需本地资源**: 云端运行

---

## 📋 准备步骤

### 1. 注册 Kaggle 账号
1. 访问 [kaggle.com](https://www.kaggle.com)
2. 注册账号并验证手机号（获取 GPU 权限）

### 2. 启用 GPU
1. 进入任意 Notebook
2. 在右侧 Settings → Accelerator → 选择 **GPU T4 x2** 或 **GPU P100**

---

## 🚀 方案一：Kaggle Notebook (推荐新手)

### Step 1: 创建新 Notebook
1. 登录 Kaggle
2. 点击 "Code" → "New Notebook"
3. 右侧设置:
   - Accelerator: **GPU T4 x2**
   - Internet: **On**
   - Persistence: **Files only**

### Step 2: 上传项目文件

在 Notebook 第一个 Cell 中：

```python
# 创建项目目录
!mkdir -p /kaggle/working/xq

# 上传方式1: 使用 Kaggle Dataset
# 先在本地把项目打包成 zip，上传为 Kaggle Dataset
# 然后在 Notebook 中添加这个 Dataset

# 上传方式2: 直接从 GitHub (如果你有仓库)
# !git clone https://github.com/your-username/xq.git /kaggle/working/xq

# 上传方式3: 使用 Kaggle 的文件上传功能
# 点击右侧 Data → Upload Dataset
```

### Step 3: 安装依赖

```python
%%bash
cd /kaggle/working/xq
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Step 4: 开始训练

```python
%%bash
cd /kaggle/working/xq

python train_rl_ai_optimized.py \
    --games 100 \
    --iterations 500 \
    --train-steps 500 \
    --batch-size 512 \
    --parallel-games 8 \
    --num-channels 256 \
    --num-res-blocks 20 \
    --save-interval 20 \
    --save-dir /kaggle/working/models
```

### Step 5: 保存模型

```python
# 训练完成后，下载模型
from IPython.display import FileLink
import os

# 列出所有模型
for f in os.listdir('/kaggle/working/models'):
    if f.endswith('.pth'):
        display(FileLink(f'/kaggle/working/models/{f}'))
```

---

## 🎯 方案二：打包上传 (推荐进阶)

### Step 1: 本地准备

在项目根目录创建 `kaggle_setup.py`:

```python
#!/usr/bin/env python3
"""
Kaggle 环境设置脚本
自动检测并配置 Kaggle 环境
"""
import os
import sys
import subprocess

def is_kaggle():
    """检测是否在 Kaggle 环境中"""
    return os.path.exists('/kaggle')

def setup_kaggle_env():
    """配置 Kaggle 环境"""
    print("检测到 Kaggle 环境，开始配置...")
    
    # 显示 GPU 信息
    print("\nGPU 信息:")
    subprocess.run(['nvidia-smi'])
    
    # 安装依赖（Kaggle 已预装 PyTorch，但可能需要更新）
    print("\n检查依赖...")
    
    # 设置工作目录
    work_dir = '/kaggle/working/xq'
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    
    print(f"\n工作目录: {work_dir}")
    print(f"输出目录: /kaggle/working/models")
    
    return work_dir

def main():
    if is_kaggle():
        work_dir = setup_kaggle_env()
        print("\n✓ Kaggle 环境配置完成！")
        print(f"\n开始训练命令:")
        print(f"cd {work_dir}")
        print(f"python train_rl_ai_optimized.py --save-dir /kaggle/working/models")
    else:
        print("未检测到 Kaggle 环境，使用本地配置")

if __name__ == "__main__":
    main()
```

### Step 2: 创建打包脚本

创建 `prepare_kaggle.py`:

```python
#!/usr/bin/env python3
"""
准备 Kaggle 上传包
"""
import os
import zipfile
import shutil

def create_kaggle_package():
    """创建 Kaggle 上传包"""
    
    # 需要包含的文件
    include_files = [
        # 核心代码
        'train_rl_ai_optimized.py',
        'kaggle_setup.py',
        
        # 模块
        'ai/__init__.py',
        'ai/ai_player.py',
        'ai/mcts_ai.py',
        
        'core/__init__.py',
        'core/game_state.py',
        'core/interfaces.py',
        
        'game/__init__.py',
        'game/dark_chess_board.py',
        'game/dark_chess_piece.py',
        'game/game_engine.py',
        'game/zobrist_hash.py',
        
        'players/__init__.py',
        'players/ai_player.py',
        'players/base_player.py',
        'players/human_player.py',
        
        'rl_ai/__init__.py',
        'rl_ai/neural_network.py',
        'rl_ai/rl_player.py',
        'rl_ai/rl_trainer.py',
        
        # 文档
        'README.md',
        'QUICK_START_OPTIMIZED.md',
    ]
    
    output_zip = 'xq_kaggle.zip'
    
    print(f"创建 Kaggle 上传包: {output_zip}")
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in include_files:
            if os.path.exists(file):
                zipf.write(file)
                print(f"  ✓ {file}")
            else:
                print(f"  ✗ {file} (不存在)")
    
    print(f"\n完成! 上传包大小: {os.path.getsize(output_zip) / 1024:.1f} KB")
    print(f"\n下一步:")
    print(f"1. 访问 kaggle.com/datasets")
    print(f"2. 点击 'New Dataset'")
    print(f"3. 上传 {output_zip}")
    print(f"4. 在 Notebook 中添加这个 Dataset")

if __name__ == "__main__":
    create_kaggle_package()
```

### Step 3: 执行打包

```bash
# 本地运行
python prepare_kaggle.py
```

### Step 4: 上传到 Kaggle

1. 访问 [kaggle.com/datasets](https://www.kaggle.com/datasets)
2. 点击 **New Dataset**
3. 上传 `xq_kaggle.zip`
4. 填写信息:
   - Title: `Chinese Dark Chess RL Training`
   - Subtitle: `揭棋 AI 训练代码`
5. 点击 **Create**

### Step 5: 在 Notebook 中使用

```python
# 新建 Notebook，添加刚上传的 Dataset

# Cell 1: 解压
!unzip -q /kaggle/input/chinese-dark-chess-rl-training/xq_kaggle.zip -d /kaggle/working/xq
!ls /kaggle/working/xq

# Cell 2: 配置环境
!cd /kaggle/working/xq && python kaggle_setup.py

# Cell 3: 开始训练
!cd /kaggle/working/xq && python train_rl_ai_optimized.py \
    --games 100 \
    --iterations 500 \
    --batch-size 512 \
    --parallel-games 8 \
    --num-channels 256 \
    --num-res-blocks 20 \
    --save-dir /kaggle/working/models \
    --save-interval 20

# Cell 4: 监控 (运行时)
!nvidia-smi
!ls -lh /kaggle/working/models/

# Cell 5: 打包下载
!cd /kaggle/working && tar -czf models_kaggle.tar.gz models/
```

---

## 🎓 方案三：使用 Kaggle Datasets 持久化

### 创建版本管理

```python
# 训练脚本中添加版本保存
import kaggle
from datetime import datetime

def save_to_kaggle_dataset(model_dir):
    """保存模型到 Kaggle Dataset"""
    version_name = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 使用 Kaggle API
    # 需要先配置 kaggle.json
    os.system(f"""
        kaggle datasets version -p {model_dir} \
        -m "Training checkpoint {version_name}"
    """)

# 在训练循环中调用
if (iteration + 1) % 50 == 0:
    save_to_kaggle_dataset('/kaggle/working/models')
```

---

## 📊 性能对比

| 环境 | GPU | 显存 | 速度 | 费用 |
|------|-----|------|------|------|
| **本地 RTX 3060** | RTX 3060 | 12GB | 100% | 电费 |
| **Kaggle T4** | Tesla T4 | 16GB | 80-90% | 免费 (30h/周) |
| **Kaggle P100** | Tesla P100 | 16GB | 120-130% | 免费 (30h/周) |
| **Google Colab** | T4/V100 | 15-16GB | 80-100% | 免费 (限时) |

💡 **建议**: Kaggle P100 比 RTX 3060 还快 20-30%！

---

## ⚠️ 注意事项

### 1. 会话限制
- Kaggle Notebook 运行时间: **12小时**
- 超时后自动停止
- 需要定期保存模型

### 2. 持久化策略

```python
# 在训练脚本中添加自动保存
import time
import signal

def setup_auto_save(trainer, save_dir):
    """设置自动保存"""
    def save_checkpoint(signum, frame):
        print("\n检测到中断，保存检查点...")
        trainer.save_checkpoint(f'{save_dir}/checkpoint_interrupt.pth')
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, save_checkpoint)
    signal.signal(signal.SIGINT, save_checkpoint)

# 使用
setup_auto_save(trainer, '/kaggle/working/models')
```

### 3. 断点续训

```python
# 检查是否有之前的模型
checkpoint_path = '/kaggle/working/models/checkpoint_interrupt.pth'
if os.path.exists(checkpoint_path):
    print(f"发现检查点，继续训练: {checkpoint_path}")
    trainer.load_checkpoint(checkpoint_path)
```

---

## 🔄 多次训练策略

### 策略 1: 分段训练
```bash
# 第一次: 训练 100 轮
python train_rl_ai_optimized.py --iterations 100 --save-dir /kaggle/working/models

# 下载 model_final.pth 到本地

# 第二次: 上传模型，继续训练
python train_rl_ai_optimized.py \
    --load-model /kaggle/input/previous-model/model_final.pth \
    --iterations 100 \
    --save-dir /kaggle/working/models
```

### 策略 2: 自动循环
```python
# 在 Notebook 中
total_iterations = 500
batch_size = 100  # 每次训练 100 轮

for i in range(0, total_iterations, batch_size):
    print(f"\n{'='*60}")
    print(f"训练批次: {i//batch_size + 1}/{total_iterations//batch_size}")
    print(f"{'='*60}\n")
    
    # 检查之前的模型
    if i > 0:
        load_model = f'/kaggle/working/models/model_iter_{i}.pth'
    else:
        load_model = None
    
    # 运行训练
    !python train_rl_ai_optimized.py \
        --load-model {load_model} \
        --iterations {batch_size} \
        --save-dir /kaggle/working/models
    
    # 保存到 Dataset（可选）
    # ... 上传代码 ...
```

---

## 📱 其他云平台选择

### Google Colab
- **优点**: 可能有 V100/A100
- **缺点**: 随机断线，不稳定
- **适合**: 短期测试

### Paperspace Gradient
- **优点**: 稳定，持久化好
- **缺点**: 免费额度少
- **适合**: 长期训练

### AWS/Azure/GCP
- **优点**: 专业，稳定
- **缺点**: 收费
- **适合**: 生产环境

---

## 🎯 推荐工作流

### 阶段 1: 本地测试 (1-2天)
```bash
# 小规模测试配置
python train_rl_ai_optimized.py --games 20 --iterations 50
```

### 阶段 2: Kaggle 训练 (1周)
- 上传代码到 Kaggle Dataset
- 创建 Notebook，每次训练 100 轮
- 每天检查 1-2 次，下载模型

### 阶段 3: 本地验证
```bash
# 下载 Kaggle 训练的模型
python main.py
# 测试 AI 强度
```

### 阶段 4: 继续优化
- 根据测试结果调整参数
- 上传新配置到 Kaggle
- 继续训练

---

## 💡 省时技巧

1. **使用快照**: 每 20 轮保存一次
2. **多账号**: 注册多个 Kaggle 账号（遵守 ToS）
3. **混合训练**: Kaggle + 本地同时进行
4. **参数调优**: 在本地小规模测试最佳参数

---

## 📚 相关资源

- [Kaggle Notebooks 文档](https://www.kaggle.com/docs/notebooks)
- [Kaggle API 文档](https://github.com/Kaggle/kaggle-api)
- [Google Colab](https://colab.research.google.com)

---

## 🚀 快速开始命令

```bash
# 1. 准备上传包
python prepare_kaggle.py

# 2. 上传到 Kaggle Dataset

# 3. 在 Kaggle Notebook 中运行
!unzip /kaggle/input/your-dataset/xq_kaggle.zip -d /kaggle/working/xq
!cd /kaggle/working/xq && python train_rl_ai_optimized.py \
    --games 100 --iterations 100 --batch-size 512 --parallel-games 8 \
    --save-dir /kaggle/working/models
```

开始在云端免费训练吧！🎉
