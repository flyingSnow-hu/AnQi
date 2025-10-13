# 揭棋游戏 - Dark Chess

一个基于Python的揭棋游戏，支持人人对战、人机对战、AI对战，并实现了基于深度强化学习的无监督AI。

## 功能特性

### 游戏模式
- 🟢 **人类 vs 人类** - 双人对战
- 🔵 **人类 vs AI** - 人机对战（简单/困难难度）
- 🟠 **AI vs AI** - 观战模式
- 🤖 **强化学习AI** - 基于神经网络的自学习AI

### AI类型
1. **SimpleAI** - 启发式AI，快速响应
2. **MCTSAI** - 蒙特卡洛树搜索AI，强度高（2000次模拟）
3. **RLPlayer** - 深度强化学习AI（无监督学习）

## 安装依赖

```bash
pip install torch numpy
```

如果要使用GPU加速训练：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 运行游戏

启动游戏界面：
```bash
python main.py
```

## 训练强化学习AI

### 快速开始（小规模训练）
```bash
python train_rl_ai.py --games 5 --iterations 10 --train-steps 50
```

### 标准训练
```bash
python train_rl_ai.py --games 10 --iterations 100 --train-steps 100 --batch-size 64
```

### 长期训练（推荐）
```bash
python train_rl_ai.py --games 50 --iterations 1000 --train-steps 200 --batch-size 128 --save-interval 50
```

### 继续训练
```bash
python train_rl_ai.py --load-model models/model_iter_100.pth --games 20 --iterations 100
```

### 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--games` | 10 | 每轮自对弈局数 |
| `--iterations` | 100 | 总训练轮数 |
| `--batch-size` | 64 | 批次大小 |
| `--train-steps` | 100 | 每轮训练步数 |
| `--save-interval` | 10 | 保存模型间隔 |
| `--load-model` | None | 加载已有模型路径 |
| `--save-dir` | models | 模型保存目录 |

## 无监督学习AI原理

### 架构设计

**神经网络模型**
- 输入：18通道特征图（10×9棋盘）
  - 通道0-6：红方明子（将、车、马、炮、士、相、兵）
  - 通道7：红方暗子
  - 通道8-14：黑方明子
  - 通道15：黑方暗子
  - 通道16：当前玩家
  - 通道17：回合进度

- 骨干网络：10层残差卷积网络
- 输出头：
  - **策略头**：8100维向量（所有可能移动的概率）
  - **价值头**：1维标量（当前局面胜率评估 -1~1）

### 训练流程

```
1. 初始化随机网络
   ↓
2. 自对弈生成训练数据
   - AI vs AI（相同网络）
   - 记录每步的状态和移动
   - 根据最终胜负标注价值
   ↓
3. 训练神经网络
   - 策略损失：交叉熵
   - 价值损失：均方误差
   - 梯度下降更新参数
   ↓
4. 重复步骤2-3
```

### 训练建议

**初期（0-100轮）**
- 网络从随机开始学习基本规则
- 胜率接近50%
- 损失逐渐下降

**中期（100-500轮）**
- 学会基本战术（吃子、保护将帅）
- 开始形成策略
- 损失趋于稳定

**后期（500+轮）**
- 掌握高级战术
- 形成独特风格
- 可以击败规则AI

**预期训练时间**
- CPU：每轮约5-10分钟（10局自对弈）
- GPU：每轮约1-2分钟

**训练1000轮大约需要**：
- CPU：50-100小时
- GPU：10-20小时

## 项目结构

```
x:\github\xq\
├── main.py                 # 游戏主程序
├── train_rl_ai.py         # RL训练脚本
├── README.md              # 说明文档
├── ai/                    # 传统AI模块
│   ├── ai_player.py       # AI基类
│   └── mcts_ai.py         # MCTS AI
├── rl_ai/                 # 强化学习模块
│   ├── neural_network.py  # 神经网络模型
│   ├── rl_trainer.py      # 训练器
│   └── rl_player.py       # RL AI玩家
├── game/                  # 游戏逻辑
│   ├── dark_chess_board.py
│   ├── dark_chess_piece.py
│   └── game_engine.py
├── ui/                    # 用户界面
│   └── dark_chess_gui.py
└── models/                # 训练模型保存目录
    ├── model_iter_10.pth
    ├── model_iter_20.pth
    └── model_final.pth
```

## 下一步开发计划

从无监督学习AI的实现开始，您可以：

### 第一步：验证框架（已完成✅）
- ✅ 创建神经网络模型
- ✅ 实现训练器
- ✅ 编写训练脚本

### 第二步：开始训练
```bash
# 先小规模测试
python train_rl_ai.py --games 2 --iterations 3 --train-steps 10

# 确认运行正常后开始正式训练
python train_rl_ai.py --games 10 --iterations 100
```

### 第三步：评估和改进
- 观察训练曲线
- 让训练好的AI与MCTS AI对战
- 调整网络结构和超参数

### 第四步：高级优化（可选）
- 实现完整的MCTS+神经网络（AlphaZero风格）
- 添加数据增强（旋转、镜像）
- 使用优先级经验回放
- 实现多进程自对弈

## 技术特点

✅ **完全无监督** - 不需要人类棋谱  
✅ **端到端学习** - 从棋盘状态直接输出策略  
✅ **自对弈进化** - AI通过与自己对弈不断进步  
✅ **可扩展架构** - 易于调整网络结构和训练策略  

## 许可证

MIT License