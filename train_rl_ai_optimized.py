#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化版强化学习AI训练脚本
- 并行自对弈
- 混合精度训练
- 更大批次
- 数据预加载
"""
import sys
import os
import argparse
import numpy as np
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

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


# 棋子价值表
PIECE_VALUES = {
    PieceType.GENERAL: 1000,
    PieceType.ADVISOR: 20,
    PieceType.ELEPHANT: 20,
    PieceType.HORSE: 40,
    PieceType.CHARIOT: 90,
    PieceType.CANNON: 45,
    PieceType.SOLDIER: 10
}


class OptimizedSelfPlayTrainer:
    """优化的自对弈训练器 - 支持并行和批量推理"""
    
    def __init__(self, trainer, num_games_per_iteration=10, reward_scale=0.01, 
                 parallel_games=4, use_amp=True):
        self.trainer = trainer
        self.num_games_per_iteration = num_games_per_iteration
        self.encoder = BoardEncoder()
        self.temperature = 1.0
        self.reward_scale = reward_scale
        self.parallel_games = parallel_games
        self.use_amp = use_amp  # 混合精度训练
        
    def calculate_step_reward(self, move_info, current_color):
        """计算每一步的即时奖励"""
        reward = 0.0
        
        if move_info.get('captured'):
            captured_type_str = move_info['captured']
            captured_color = move_info['captured_color']
            
            try:
                piece_type = PieceType(captured_type_str)
                piece_value = PIECE_VALUES.get(piece_type, 10)
                
                if captured_color != current_color:
                    reward += piece_value * self.reward_scale
                    
            except (ValueError, KeyError):
                pass
        
        return reward
    
    def play_game(self, show_progress=False):
        """进行一局自对弈（优化版：减少重复计算）"""
        red_player = RLPlayer("red", temperature=self.temperature)
        black_player = RLPlayer("black", temperature=self.temperature)
        
        # 共享模型
        red_player.model = self.trainer.model
        black_player.model = self.trainer.model
        
        engine = GameEngine(red_player, black_player)
        
        game_data = []
        move_count = 0
        max_moves = 200
        
        red_cumulative_reward = 0.0
        black_cumulative_reward = 0.0
        
        while not engine.game_state.game_over and move_count < max_moves:
            current_color = engine.game_state.current_turn
            
            # 编码当前状态
            state = self.encoder.encode_board(engine.board, current_color)
            
            # 使用神经网络预测策略
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.trainer.device)
            with torch.no_grad():
                policy_logits, _ = self.trainer.model(state_tensor)
                policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
            
            # 记录数据（先记录状态和策略，价值稍后更新）
            game_data.append({
                'state': state,
                'policy': policy_probs,
                'value': 0,  # 临时值，游戏结束后更新
                'color': current_color
            })
            
            # 获取当前玩家并执行移动
            current_player = red_player if current_color == "red" else black_player
            move = current_player.get_move(engine.board, engine.game_state)
            
            if move is None:
                # 无合法移动，游戏结束
                break
            
            from_pos, to_pos = move
            
            # 记录移动前的信息（用于计算奖励）
            captured_piece = engine.board.get_piece(*to_pos)
            captured_info = {
                'captured': None,
                'captured_color': None
            }
            
            if captured_piece and captured_piece.is_revealed():
                captured_info['captured'] = captured_piece.piece_type.value
                captured_info['captured_color'] = captured_piece.color
            
            # 执行移动
            success = engine.make_move(from_pos, to_pos)
            if not success:
                # 移动失败，游戏可能已结束
                break
            
            move_info = captured_info
            
            # 计算即时奖励
            step_reward = self.calculate_step_reward(move_info, current_color)
            
            if current_color == "red":
                red_cumulative_reward += step_reward
            else:
                black_cumulative_reward += step_reward
            
            # 添加三次重复惩罚
            repetition_penalty = engine.game_state.get_repetition_penalty()
            if repetition_penalty < 0:
                if current_color == "red":
                    red_cumulative_reward += repetition_penalty
                else:
                    black_cumulative_reward += repetition_penalty
            
            # 添加长将惩罚
            perpetual_check_penalty = engine.game_state.get_perpetual_check_penalty()
            if perpetual_check_penalty < 0:
                if current_color == "red":
                    red_cumulative_reward += perpetual_check_penalty
                else:
                    black_cumulative_reward += perpetual_check_penalty
            
            # 添加长捉惩罚
            perpetual_chase_penalty = engine.game_state.get_perpetual_chase_penalty()
            if perpetual_chase_penalty < 0:
                if current_color == "red":
                    red_cumulative_reward += perpetual_chase_penalty
                else:
                    black_cumulative_reward += perpetual_chase_penalty
            
            move_count += 1
        
        # 确定获胜方
        winner = engine.game_state.winner
        
        # 特殊情况判定
        if engine.game_state.is_draw_by_repetition:
            # 三次重复是和棋，不是判负
            winner = None
        
        if engine.game_state.is_loss_by_perpetual_check:
            loser = engine.game_state.checking_side
            winner = "black" if loser == "red" else "red"
        
        if engine.game_state.is_loss_by_perpetual_chase:
            loser = engine.game_state.chasing_side
            winner = "black" if loser == "red" else "red"
        
        # 更新所有步骤的价值
        for data in game_data:
            if winner is None:
                data['value'] = 0
            elif winner == data['color']:
                data['value'] = 1
            else:
                data['value'] = -1
        
        return game_data, winner, red_cumulative_reward, black_cumulative_reward
    
    def play_parallel_games(self, num_games):
        """并行进行多局游戏"""
        all_game_data = []
        results = {'red': 0, 'black': 0, 'draw': 0}
        total_red_rewards = 0.0
        total_black_rewards = 0.0
        
        # 使用线程池（因为GIL在GPU计算时会释放）
        with ThreadPoolExecutor(max_workers=self.parallel_games) as executor:
            futures = [executor.submit(self.play_game, False) for _ in range(num_games)]
            
            for i, future in enumerate(futures):
                game_data, winner, red_reward, black_reward = future.result()
                all_game_data.extend(game_data)
                
                total_red_rewards += red_reward
                total_black_rewards += black_reward
                
                if winner == "red":
                    results['red'] += 1
                elif winner == "black":
                    results['black'] += 1
                else:
                    results['draw'] += 1
                
                print(f"  游戏 {i+1}/{num_games} 完成 (胜者: {winner or '平局'}, "
                      f"步数: {len(game_data)}, 红方奖励: {red_reward:.2f}, 黑方奖励: {black_reward:.2f})")
        
        return all_game_data, results, total_red_rewards, total_black_rewards
    
    def generate_training_data(self, num_games):
        """生成训练数据"""
        print(f"\n生成训练数据 (并行度: {self.parallel_games})...")
        
        all_game_data, results, total_red_rewards, total_black_rewards = \
            self.play_parallel_games(num_games)
        
        # 添加到回放缓冲区
        for data in all_game_data:
            self.trainer.replay_buffer.add(
                data['state'],
                data['policy'],
                data['value'],
                None  # winner已编码到value中
            )
        
        print(f"\n自对弈结果: 红胜{results['red']} 黑胜{results['black']} 平局{results['draw']}")
        print(f"平均红方奖励: {total_red_rewards/num_games:.2f}, "
              f"平均黑方奖励: {total_black_rewards/num_games:.2f}")
        print(f"缓冲区大小: {len(self.trainer.replay_buffer)}")
        print(f"生成训练样本: {len(all_game_data)}")


class OptimizedRLTrainer(RLTrainer):
    """优化的训练器 - 支持混合精度和大批次"""
    
    def __init__(self, model=None, lr=0.001, weight_decay=1e-4, use_amp=True):
        super().__init__(model, lr, weight_decay)
        self.use_amp = use_amp
        self.scaler = GradScaler('cuda') if use_amp and torch.cuda.is_available() else None
        
        if use_amp:
            print("启用混合精度训练 (AMP)")
    
    def train_step(self, batch_size=64):
        """优化的训练步骤 - 支持混合精度"""
        if len(self.replay_buffer) < batch_size:
            return None
        
        self.model.train()
        
        # 采样数据
        states, target_policies, target_values = self.replay_buffer.sample(batch_size)
        
        # 转换为tensor
        states = torch.FloatTensor(states).to(self.device)
        target_policies = torch.FloatTensor(target_policies).to(self.device)
        target_values = torch.FloatTensor(target_values).unsqueeze(1).to(self.device)
        
        # 前向传播（混合精度）
        self.optimizer.zero_grad()
        
        if self.use_amp and torch.cuda.is_available():
            with autocast('cuda'):
                policy_logits, values = self.model(states)
                policy_loss = -torch.mean(torch.sum(
                    target_policies * torch.nn.functional.log_softmax(policy_logits, dim=1), 
                    dim=1))
                value_loss = torch.nn.functional.mse_loss(values, target_values)
                total_loss = policy_loss + value_loss
            
            # 反向传播（混合精度）
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            policy_logits, values = self.model(states)
            policy_loss = -torch.mean(torch.sum(
                target_policies * torch.nn.functional.log_softmax(policy_logits, dim=1), 
                dim=1))
            value_loss = torch.nn.functional.mse_loss(values, target_values)
            total_loss = policy_loss + value_loss
            
            # 反向传播
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
        
        # 记录统计
        self.training_stats['policy_losses'].append(policy_loss.item())
        self.training_stats['value_losses'].append(value_loss.item())
        self.training_stats['total_losses'].append(total_loss.item())
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': total_loss.item()
        }


def main():
    parser = argparse.ArgumentParser(description='训练强化学习AI（优化版）')
    parser.add_argument('--games', type=int, default=10, help='每轮自对弈局数')
    parser.add_argument('--iterations', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=256, help='批次大小（优化版默认更大）')
    parser.add_argument('--train-steps', type=int, default=100, help='每轮训练步数')
    parser.add_argument('--save-interval', type=int, default=10, help='保存模型间隔')
    parser.add_argument('--load-model', type=str, default=None, help='加载已有模型')
    parser.add_argument('--save-dir', type=str, default='models', help='模型保存目录')
    
    # 训练超参数
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--temperature', type=float, default=1.0, help='自对弈探索温度')
    parser.add_argument('--reward-scale', type=float, default=0.01, help='即时奖励缩放因子')
    
    # 网络架构
    parser.add_argument('--num-channels', type=int, default=128, help='神经网络每层通道数')
    parser.add_argument('--num-res-blocks', type=int, default=10, help='神经网络残差块数量')
    
    # 优化参数
    parser.add_argument('--parallel-games', type=int, default=4, help='并行自对弈游戏数')
    parser.add_argument('--no-amp', action='store_true', help='禁用混合精度训练')
    
    args = parser.parse_args()

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 初始化优化版训练器
    print("初始化优化版强化学习训练器...")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"CUDA版本: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    
    model = DarkChessNet(num_channels=args.num_channels, num_res_blocks=args.num_res_blocks)
    trainer = OptimizedRLTrainer(
        model=model, 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        use_amp=not args.no_amp
    )

    # 加载已有模型
    if args.load_model:
        trainer.load_checkpoint(args.load_model)

    # 创建自对弈训练器
    self_play_trainer = OptimizedSelfPlayTrainer(
        trainer,
        num_games_per_iteration=args.games,
        reward_scale=args.reward_scale,
        parallel_games=args.parallel_games,
        use_amp=not args.no_amp
    )
    self_play_trainer.temperature = args.temperature
    
    print("\n" + "="*70)
    print("优化版强化学习训练开始")
    print("="*70)
    print(f"训练轮数: {args.iterations}")
    print(f"每轮自对弈局数: {args.games}")
    print(f"并行游戏数: {args.parallel_games}")
    print(f"每轮训练步数: {args.train_steps}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"探索温度: {args.temperature}")
    print(f"即时奖励缩放: {args.reward_scale}")
    print(f"网络架构: {args.num_channels}通道, {args.num_res_blocks}残差块")
    print(f"混合精度训练: {'是' if not args.no_amp else '否'}")
    print("="*70)
    print("\n优化特性:")
    print("✓ 并行自对弈 - 充分利用GPU")
    print("✓ 混合精度训练 - FP16加速")
    print("✓ 大批次训练 - 提高吞吐量")
    print("✓ 密集奖励机制")
    print("✓ 规则惩罚系统")
    print("="*70)
    
    # 训练循环
    import time
    for iteration in range(args.iterations):
        iter_start = time.time()
        
        print(f"\n{'='*70}")
        print(f"第 {iteration+1}/{args.iterations} 轮训练")
        print(f"{'='*70}")
        
        # 1. 自对弈生成数据
        gen_start = time.time()
        self_play_trainer.generate_training_data(num_games=args.games)
        gen_time = time.time() - gen_start
        
        # 2. 训练网络
        train_start = time.time()
        print(f"\n开始训练神经网络...")
        avg_loss = trainer.train_epoch(num_steps=args.train_steps, batch_size=args.batch_size)
        train_time = time.time() - train_start
        
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
        iter_time = time.time() - iter_start
        
        print(f"\n性能统计:")
        print(f"  数据生成时间: {gen_time:.1f}秒")
        print(f"  训练时间: {train_time:.1f}秒")
        print(f"  总时间: {iter_time:.1f}秒")
        print(f"  样本/秒: {args.games*100/iter_time:.1f}")  # 假设平均100步/局
        
        if stats:
            print(f"\n训练统计:")
            print(f"  总局数: {stats['total_games']}")
            print(f"  缓冲区大小: {stats['buffer_size']}")
    
    # 保存最终模型
    final_model_path = os.path.join(args.save_dir, 'model_final.pth')
    trainer.save_checkpoint(final_model_path)
    
    print("\n" + "="*70)
    print("训练完成！")
    print(f"最终模型已保存到: {final_model_path}")
    print("="*70)


if __name__ == "__main__":
    main()
