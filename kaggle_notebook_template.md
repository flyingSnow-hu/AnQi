# 揭棋 AI 训练 - Kaggle Notebook

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
!cd /kaggle/working && python train_rl_ai_optimized.py \
    --games 10 \
    --iterations 5 \
    --train-steps 50 \
    --batch-size 256 \
    --parallel-games 4 \
    --num-channels 128 \
    --num-res-blocks 10 \
    --save-dir /kaggle/working/models_test
```

## 4. 正式训练 (高性能配置)

```python
# 这将运行较长时间，建议在确认环境正常后再运行
!cd /kaggle/working && python train_rl_ai_optimized.py \
    --games 100 \
    --iterations 500 \
    --train-steps 500 \
    --batch-size 512 \
    --parallel-games 8 \
    --num-channels 256 \
    --num-res-blocks 20 \
    --save-dir /kaggle/working/models \
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
    print("\n已保存的模型:")
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
!cd /kaggle/working && python train_rl_ai_optimized.py \
    --load-model /kaggle/working/models/model_iter_100.pth \
    --games 100 \
    --iterations 400 \
    --batch-size 512 \
    --parallel-games 8 \
    --save-dir /kaggle/working/models
```

## 提示

- Kaggle Notebook 最长运行 12 小时
- 每 20 轮会自动保存模型
- 建议定期下载模型到本地备份
- GPU T4: 适合批次 256-512
- GPU P100: 适合批次 512-1024
