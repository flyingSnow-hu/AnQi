# 强化学习AI训练指南

## 🎯 训练参数详解

### 基础参数
- **--games**: 每轮自对弈局数
  - 更多局数 → 更多样化的训练数据
  - 推荐: 10-50局
  
- **--iterations**: 训练轮数
  - 总训练轮次
  - 推荐: 100-1000轮
  
- **--train-steps**: 每轮训练步数
  - 每轮从回放缓冲区采样训练的次数
  - 推荐: 100-500步

- **--batch-size**: 批次大小
  - 每次训练使用的样本数量
  - 推荐: 32-128（取决于GPU显存）

### 高级参数
- **--lr**: 学习率
  - 控制模型参数更新的步长
  - 默认: 0.001
  - 降低学习率 → 训练更稳定但更慢
  - 推荐范围: 0.0001 - 0.01

- **--weight-decay**: 权重衰减
  - L2正则化强度，防止过拟合
  - 默认: 1e-4
  - 推荐范围: 1e-5 - 1e-3

- **--temperature**: 探索温度
  - 控制自对弈时的探索程度
  - 0.0 = 完全贪心（选最优）
  - 1.0 = 随机探索（按概率采样）
  - 训练初期建议1.0，后期可降到0.5

## 📊 推荐训练方案

### 方案1: 快速测试（1-2小时）
适合验证训练流程是否正常
```bash
python train_rl_ai.py \
  --load-model models/model_final.pth \
  --games 20 \
  --iterations 50 \
  --train-steps 150 \
  --batch-size 64 \
  --lr 0.001 \
  --temperature 0.8 \
  --save-interval 10
```

### 方案2: 中等训练（6-12小时）⭐推荐
适合提升AI棋力到中级水平
```bash
python train_rl_ai.py \
  --load-model models/model_final.pth \
  --games 30 \
  --iterations 200 \
  --train-steps 200 \
  --batch-size 64 \
  --lr 0.0005 \
  --temperature 0.7 \
  --save-interval 20 \
  --save-dir models_v2
```

### 方案3: 深度训练（24-48小时）
适合训练高水平AI
```bash
python train_rl_ai.py \
  --load-model models/model_final.pth \
  --games 50 \
  --iterations 500 \
  --train-steps 300 \
  --batch-size 128 \
  --lr 0.0003 \
  --temperature 0.6 \
  --save-interval 25 \
  --save-dir models_advanced
```

### 方案4: 极致训练（数天）
追求最强AI
```bash
python train_rl_ai.py \
  --load-model models/model_final.pth \
  --games 100 \
  --iterations 1000 \
  --train-steps 500 \
  --batch-size 128 \
  --lr 0.0001 \
  --temperature 0.5 \
  --save-interval 50 \
  --save-dir models_master
```

## 🔧 参数调优技巧

### 提升棋力的关键参数

1. **增加训练数据多样性**
   ```bash
   --games 50          # 更多自对弈局数
   --temperature 0.8   # 保持一定探索
   ```

2. **更深入的学习**
   ```bash
   --iterations 500    # 更多训练轮次
   --train-steps 300   # 每轮更多训练步数
   ```

3. **稳定训练（避免过拟合）**
   ```bash
   --lr 0.0003         # 降低学习率
   --weight-decay 1e-4 # 适度正则化
   ```

4. **GPU显存优化**
   - GTX 1050 (2-4GB): --batch-size 32-64
   - RTX 3060 (12GB): --batch-size 128-256
   - RTX 4090 (24GB): --batch-size 256-512

## 📈 训练监控

### 观察指标
- **策略损失** (policy_loss): 应逐渐降低
  - 初期: 6-8
  - 中期: 3-5
  - 后期: 1-3

- **价值损失** (value_loss): 应逐渐降低
  - 初期: 0.3-0.5
  - 中期: 0.1-0.3
  - 后期: 0.05-0.15

- **自对弈胜率**: 红黑应接近50%
  - 严重偏向一方 → 可能存在问题

### 异常情况处理

1. **损失不下降**
   - 降低学习率: --lr 0.0001
   - 增加批次大小: --batch-size 128
   - 检查是否需要更多数据

2. **损失震荡**
   - 降低学习率: --lr 0.0003
   - 降低探索温度: --temperature 0.5

3. **过拟合（训练损失低但棋力未提升）**
   - 增加权重衰减: --weight-decay 1e-3
   - 增加自对弈局数: --games 50
   - 提高探索温度: --temperature 0.9

## 💡 实战建议

### 阶段性训练策略

**阶段1: 探索期（0-100轮）**
- 目标: 学习基本规则和策略
- 参数: --games 30, --lr 0.001, --temperature 1.0

**阶段2: 强化期（100-300轮）**
- 目标: 优化策略，提升战术
- 参数: --games 40, --lr 0.0005, --temperature 0.7

**阶段3: 精炼期（300-500轮）**
- 目标: 微调，追求极致
- 参数: --games 50, --lr 0.0003, --temperature 0.5

**阶段4: 稳定期（500+轮）**
- 目标: 保持水平，防止退化
- 参数: --games 30, --lr 0.0001, --temperature 0.3

### 测试AI水平

定期测试当前模型：
```bash
# 启动游戏GUI
python main.py

# 选择 "MCTS AI vs RL AI" 观战
# 如果RL AI胜率>50%，说明已超越MCTS
```

## 🎮 使用训练好的模型

训练完成后，模型自动保存：
- `models/model_final.pth` - 最终模型
- `models/model_iter_XX.pth` - 中间检查点

在游戏GUI中选择"人类 vs RL AI"即可对战！

---

**提示**: 训练是个持续过程，建议从中等训练方案开始，观察效果后再决定是否深度训练！
