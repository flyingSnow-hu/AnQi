#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进版强化学习AI训练脚本
增加密集奖励机制，提升AI对吃子和躲避被吃的意识
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
from game.dark_chess_piece import PieceType
from game.zobrist_hash import ZobristHash
from players.human_player import HumanPlayer
from rl_ai.neural_network import DarkChessNet, BoardEncoder
from rl_ai.rl_trainer import RLTrainer
from rl_ai.rl_player import RLPlayer


# 棋子价值表（用于计算吃子奖励）
PIECE_VALUES = {
    PieceType.GENERAL: 1000,   # 将/帅
    PieceType.ADVISOR: 20,     # 士
    PieceType.ELEPHANT: 20,    # 象
    PieceType.HORSE: 40,       # 马
    PieceType.CHARIOT: 90,     # 车
    PieceType.CANNON: 45,      # 炮
    PieceType.SOLDIER: 10      # 兵/卒
}


class ImprovedSelfPlayTrainer:
    """改进的自对弈训练器 - 包含密集奖励"""
    
    def __init__(self, trainer, num_games_per_iteration=10, reward_scale=0.01):
        self.trainer = trainer
        self.num_games_per_iteration = num_games_per_iteration
        self.encoder = BoardEncoder()
        self.temperature = 1.0
        self.reward_scale = reward_scale  # 即时奖励缩放因子
        
    def calculate_step_reward(self, move_info, current_color):
        """
        计算每一步的即时奖励
        move_info: 移动信息字典（包含captured等）
        current_color: 当前玩家颜色
        """
        reward = 0.0
        
        # 吃子奖励
        if move_info.get('captured'):
            captured_type_str = move_info['captured']
            captured_color = move_info['captured_color']
            
            # 将字符串转换为PieceType枚举
            try:
                piece_type = PieceType(captured_type_str)
                piece_value = PIECE_VALUES.get(piece_type, 10)
                
                # 如果吃掉对方的棋子，给予正奖励
                if captured_color != current_color:
                    reward += piece_value * self.reward_scale
                    
            except (ValueError, KeyError):
                pass
        
        return reward
    
    def play_game(self, show_progress=False):
        """
        进行一局自对弈（改进版：记录即时奖励）
        返回: (game_data, winner)
        """
        # 创建两个AI玩家
        red_player = RLPlayer("red", temperature=self.temperature)
        black_player = RLPlayer("black", temperature=self.temperature)
        
        # 共享相同的模型
        red_player.model = self.trainer.model
        black_player.model = self.trainer.model
        
        # 创建游戏引擎
        engine = GameEngine(red_player, black_player)
        
        game_data = []
        move_count = 0
        max_moves = 200
        
        # 记录每一步的累积奖励
        red_cumulative_reward = 0.0
        black_cumulative_reward = 0.0
        
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
            
            # 记录移动策略
            policy = np.zeros(10 * 9 * 10 * 9, dtype=np.float32)
            move_idx = self.encoder.encode_move(move[0], move[1])
            policy[move_idx] = 1.0
            
            # 执行移动并获取移动信息
            from_pos, to_pos = move
            move_info = {
                'from': from_pos,
                'to': to_pos,
                'color': current_color
            }
            
            # 记录移动前的状态（用于计算奖励）
            old_captured_count = len(engine.game_state.captured_pieces.get(current_color, []))
            
            # 执行移动
            captured_piece = engine.board.move_piece(from_pos, to_pos)
            
            # 更新游戏状态
            if captured_piece:
                move_info['captured'] = captured_piece.piece_type.value
                move_info['captured_color'] = captured_piece.color
                engine.game_state.captured_pieces[captured_piece.color].append(captured_piece)
            
            engine.game_state.move_count += 1
            
            # 检查游戏是否结束
            if engine.board.is_general_captured("red"):
                engine.game_state.game_over = True
                engine.game_state.winner = "黑方"
            elif engine.board.is_general_captured("black"):
                engine.game_state.game_over = True
                engine.game_state.winner = "红方"
            
            # 切换回合
            engine.game_state.current_turn = "black" if current_color == "red" else "red"
            
            # 计算新局面哈希并检查重复
            current_hash = engine.zobrist.compute_hash(engine.board, engine.game_state.current_turn)
            is_draw = engine.game_state.add_position_hash(current_hash)
            
            # 如果三次重复,标记为平局
            if is_draw:
                engine.game_state.game_over = True
                engine.game_state.winner = None
            
            # 计算即时奖励(包含吃子奖励和重复惩罚)
            step_reward = self.calculate_step_reward(move_info, current_color)
            
            # 添加重复局面惩罚
            repetition_penalty = engine.game_state.get_repetition_penalty()
            step_reward += repetition_penalty
            
            # 添加长将惩罚
            perpetual_check_penalty = engine.game_state.get_perpetual_check_penalty()
            step_reward += perpetual_check_penalty
            
            # 累积奖励
            if current_color == "red":
                red_cumulative_reward += step_reward
            else:
                black_cumulative_reward += step_reward
            
            # 保存状态和策略（价值稍后根据游戏结果填充）
            game_data.append({
                'state': board_state,
                'policy': policy,
                'color': current_color,
                'step_reward': step_reward  # 记录即时奖励
            })
            
            move_count += 1
            
            if show_progress and move_count % 10 == 0:
                print(f"  第{move_count}步", end='\r')
        
        # 确定胜负
        if engine.game_state.game_over:
            if engine.game_state.is_draw_by_repetition:
                winner = None  # 三次重复平局
            elif engine.game_state.is_loss_by_perpetual_check:
                # 长将判负
                if engine.game_state.winner == "红方":
                    winner = "red"
                elif engine.game_state.winner == "黑方":
                    winner = "black"
                else:
                    winner = None
            elif engine.game_state.winner == "红方":
                winner = "red"
            elif engine.game_state.winner == "黑方":
                winner = "black"
            else:
                winner = None  # 其他平局情况
        else:
            winner = None  # 超过最大步数,平局
        
        # 填充价值（结合最终胜负和即时奖励）
        for data in game_data:
            # 基础价值（根据最终胜负）
            if winner is None:
                final_value = 0.0  # 平局
            elif data['color'] == winner:
                final_value = 1.0  # 胜
            else:
                final_value = -1.0  # 负
            
            # 结合即时奖励
            # 即时奖励的影响会随着时间衰减（越接近游戏结束，最终胜负的影响越大）
            data['value'] = final_value + data['step_reward']
        
        return game_data, winner, red_cumulative_reward, black_cumulative_reward
    
    def generate_training_data(self, num_games=10):
        """生成训练数据"""
        print(f"\n开始自对弈，生成 {num_games} 局训练数据...")
        
        red_wins = 0
        black_wins = 0
        draws = 0
        total_red_rewards = 0.0
        total_black_rewards = 0.0
        
        for i in range(num_games):
            print(f"\n第 {i+1}/{num_games} 局自对弈...", end='')
            game_data, winner, red_reward, black_reward = self.play_game(show_progress=True)
            
            total_red_rewards += red_reward
            total_black_rewards += black_reward
            
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
            
            print(f" 完成 (胜者: {winner or '平局'}, 步数: {len(game_data)}, 红方奖励: {red_reward:.2f}, 黑方奖励: {black_reward:.2f})")
        
        print(f"\n自对弈结果: 红胜{red_wins} 黑胜{black_wins} 平局{draws}")
        print(f"平均红方奖励: {total_red_rewards/num_games:.2f}, 平均黑方奖励: {total_black_rewards/num_games:.2f}")
        print(f"缓冲区大小: {len(self.trainer.replay_buffer)}")


def main():
    parser = argparse.ArgumentParser(description='训练强化学习AI（改进版）')
    parser.add_argument('--games', type=int, default=10, help='每轮自对弈局数')
    parser.add_argument('--iterations', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=64, help='批次大小')
    parser.add_argument('--train-steps', type=int, default=100, help='每轮训练步数')
    parser.add_argument('--save-interval', type=int, default=10, help='保存模型间隔')
    parser.add_argument('--load-model', type=str, default=None, help='加载已有模型')
    parser.add_argument('--save-dir', type=str, default='models', help='模型保存目录')
    
    # 训练超参数
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--temperature', type=float, default=1.0, help='自对弈探索温度')
    parser.add_argument('--reward-scale', type=float, default=0.01, help='即时奖励缩放因子')
    
    parser.add_argument('--num-channels', type=int, default=128, help='神经网络每层通道数 (宽度)')
    parser.add_argument('--num-res-blocks', type=int, default=10, help='神经网络残差块数量 (深度)')
    args = parser.parse_args()

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 初始化自定义规模的神经网络
    print("初始化改进版强化学习训练器...")
    model = DarkChessNet(num_channels=args.num_channels, num_res_blocks=args.num_res_blocks)
    trainer = RLTrainer(model=model, lr=args.lr, weight_decay=args.weight_decay)

    # 加载已有模型
    if args.load_model:
        trainer.load_checkpoint(args.load_model)

    # 创建自对弈训练器
    self_play_trainer = ImprovedSelfPlayTrainer(
        trainer,
        num_games_per_iteration=args.games,
        reward_scale=args.reward_scale
    )
    self_play_trainer.temperature = args.temperature
    
    print("\n" + "="*60)
    print("改进版强化学习训练开始（密集奖励机制）")
    print("="*60)
    print(f"训练轮数: {args.iterations}")
    print(f"每轮自对弈局数: {args.games}")
    print(f"每轮训练步数: {args.train_steps}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"探索温度: {args.temperature}")
    print(f"即时奖励缩放: {args.reward_scale}")
    print("="*60)
    print("\n特性:")
    print("✓ 吃子即时奖励")
    print("✓ 根据棋子价值分配不同奖励")
    print("✓ 帮助AI学习战术意识")
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
