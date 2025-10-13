#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强化学习AI训练脚本
通过自对弈生成训练数据，训练神经网络
"""
import sys
import os
import argparse
import numpy as np
from datetime import datetime

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from game.game_engine import GameEngine
from players.human_player import HumanPlayer
from rl_ai.neural_network import DarkChessNet, BoardEncoder
from rl_ai.rl_trainer import RLTrainer
from rl_ai.rl_player import RLPlayer


class SelfPlayTrainer:
    """自对弈训练器"""
    
    def __init__(self, trainer, num_games_per_iteration=10):
        self.trainer = trainer
        self.num_games_per_iteration = num_games_per_iteration
        self.encoder = BoardEncoder()
        self.temperature = 1.0  # 默认探索温度
        
    def play_game(self, show_progress=False):
        """
        进行一局自对弈
        返回: (game_data, winner)
        game_data: [(state, policy, value), ...]
        """
        # 创建两个使用相同网络的AI玩家
        red_player = RLPlayer("red", temperature=self.temperature)  # 使用配置的温度
        black_player = RLPlayer("black", temperature=self.temperature)
        
        # 共享相同的模型
        red_player.model = self.trainer.model
        black_player.model = self.trainer.model
        
        # 创建游戏引擎
        engine = GameEngine(red_player, black_player)
        
        game_data = []
        move_count = 0
        max_moves = 200
        
        while not engine.game_state.game_over and move_count < max_moves:
            # 记录当前状态
            current_color = engine.game_state.current_turn
            board_state = self.encoder.encode_board(engine.board, current_color)
            
            # 获取AI移动
            if current_color == "red":
                move = red_player.get_move(engine.board, engine.game_state)
            else:
                move = black_player.get_move(engine.board, engine.game_state)
            
            if not move:
                break
            
            # 记录移动策略（简化版：one-hot编码）
            policy = np.zeros(10 * 9 * 10 * 9, dtype=np.float32)
            move_idx = self.encoder.encode_move(move[0], move[1])
            policy[move_idx] = 1.0
            
            # 保存状态和策略（价值稍后根据游戏结果填充）
            game_data.append({
                'state': board_state,
                'policy': policy,
                'color': current_color
            })
            
            # 执行移动
            engine.make_move(move[0], move[1])
            move_count += 1
            
            if show_progress and move_count % 10 == 0:
                print(f"  第{move_count}步", end='\r')
        
        # 确定胜负
        if engine.game_state.game_over:
            winner = "red" if engine.game_state.winner == "红方" else "black"
        else:
            winner = None  # 平局
        
        # 填充价值
        for data in game_data:
            if winner is None:
                data['value'] = 0.0  # 平局
            elif data['color'] == winner:
                data['value'] = 1.0  # 胜
            else:
                data['value'] = -1.0  # 负
        
        return game_data, winner
    
    def generate_training_data(self, num_games=10):
        """生成训练数据"""
        print(f"\n开始自对弈，生成 {num_games} 局训练数据...")
        
        red_wins = 0
        black_wins = 0
        draws = 0
        
        for i in range(num_games):
            print(f"\n第 {i+1}/{num_games} 局自对弈...", end='')
            game_data, winner = self.play_game(show_progress=True)
            
            # 统计胜负
            if winner == "red":
                red_wins += 1
            elif winner == "black":
                black_wins += 1
            else:
                draws += 1
            
            # 添加到回放缓冲区
            for data in game_data:
                self.trainer.replay_buffer.add(
                    data['state'],
                    data['policy'],
                    data['value'],
                    winner
                )
            
            print(f" 完成 (胜者: {winner or '平局'}, 步数: {len(game_data)})")
        
        print(f"\n自对弈结果: 红胜{red_wins} 黑胜{black_wins} 平局{draws}")
        print(f"缓冲区大小: {len(self.trainer.replay_buffer)}")


def main():
    parser = argparse.ArgumentParser(description='训练强化学习AI')
    parser.add_argument('--games', type=int, default=10, help='每轮自对弈局数')
    parser.add_argument('--iterations', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=64, help='批次大小')
    parser.add_argument('--train-steps', type=int, default=100, help='每轮训练步数')
    parser.add_argument('--save-interval', type=int, default=10, help='保存模型间隔')
    parser.add_argument('--load-model', type=str, default=None, help='加载已有模型')
    parser.add_argument('--save-dir', type=str, default='models', help='模型保存目录')
    
    # 新增：训练超参数
    parser.add_argument('--lr', type=float, default=0.001, help='学习率 (默认0.001，降低可稳定训练)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='权重衰减 (正则化强度)')
    parser.add_argument('--temperature', type=float, default=1.0, help='自对弈探索温度 (0=贪心, 1=随机探索)')
    
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 初始化训练器
    print("初始化强化学习训练器...")
    trainer = RLTrainer(lr=args.lr, weight_decay=args.weight_decay)
    
    # 加载已有模型
    if args.load_model:
        trainer.load_checkpoint(args.load_model)
    
    # 创建自对弈训练器
    self_play_trainer = SelfPlayTrainer(trainer, num_games_per_iteration=args.games)
    self_play_trainer.temperature = args.temperature  # 设置探索温度
    
    print("\n" + "="*60)
    print("强化学习训练开始")
    print("="*60)
    print(f"训练轮数: {args.iterations}")
    print(f"每轮自对弈局数: {args.games}")
    print(f"每轮训练步数: {args.train_steps}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"探索温度: {args.temperature}")
    print("="*60)
    
    # 训练循环
    for iteration in range(args.iterations):
        print(f"\n{'='*60}")
        print(f"第 {iteration+1}/{args.iterations} 轮训练")
        print(f"{'='*60}")
        
        # 1. 自对弈生成数据
        self_play_trainer.generate_training_data(num_games=args.games)
        
        # 2. 训练网络
        print(f"\n开始训练神经网络...")
        avg_loss = trainer.train_epoch(num_steps=args.train_steps, batch_size=args.batch_size)
        
        if avg_loss:
            print(f"训练损失:")
            print(f"  策略损失: {avg_loss['policy_loss']:.4f}")
            print(f"  价值损失: {avg_loss['value_loss']:.4f}")
            print(f"  总损失: {avg_loss['total_loss']:.4f}")
        
        # 3. 保存模型
        if (iteration + 1) % args.save_interval == 0:
            model_path = os.path.join(args.save_dir, f'model_iter_{iteration+1}.pth')
            trainer.save_checkpoint(model_path, epoch=iteration+1)
        
        # 4. 显示统计
        stats = trainer.get_stats()
        if stats:
            print(f"\n训练统计:")
            print(f"  总局数: {stats['total_games']}")
            print(f"  缓冲区大小: {stats['buffer_size']}")
    
    # 保存最终模型
    final_model_path = os.path.join(args.save_dir, 'model_final.pth')
    trainer.save_checkpoint(final_model_path)
    
    print("\n" + "="*60)
    print("训练完成！")
    print(f"最终模型已保存到: {final_model_path}")
    print("="*60)


if __name__ == "__main__":
    main()
