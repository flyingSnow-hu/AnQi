"""
基于神经网络的强化学习AI玩家
使用训练好的神经网络进行决策
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import Tuple, List, Optional
import torch

from ai.ai_player import AIPlayer
from .neural_network import DarkChessNet, BoardEncoder


class RLPlayer(AIPlayer):
    """基于神经网络的强化学习AI"""
    
    def __init__(self, color: str, model_path=None, temperature=0.1):
        """
        初始化RL玩家
        color: AI颜色 ("red" or "black")
        model_path: 模型文件路径，如果为None则使用随机初始化的模型
        temperature: 温度参数，控制探索程度（0=贪心，1=完全随机）
        """
        super().__init__(color)
        self.name = f"RL AI ({color})"
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DarkChessNet(num_channels=128, num_res_blocks=10)
        self.model.to(self.device)
        
        # 加载模型
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"RL AI ({color}) 已加载模型: {model_path}")
        else:
            print(f"RL AI ({color}) 使用随机初始化模型")
        
        self.model.eval()
        self.temperature = temperature
        self.encoder = BoardEncoder()
    
    def get_move(self, board, game_state) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        获取AI的移动决策
        使用神经网络预测策略，然后选择最佳合法移动
        """
        legal_moves = self.get_all_legal_moves(board, self.color)
        
        if not legal_moves:
            return None
        
        if len(legal_moves) == 1:
            return legal_moves[0]
        
        # 立即吃将
        for from_pos, to_pos in legal_moves:
            target = board.get_piece(*to_pos)
            if target and hasattr(target, 'piece_type'):
                from game.dark_chess_piece import PieceType
                if target.piece_type == PieceType.GENERAL:
                    return (from_pos, to_pos)
        
        # 使用神经网络预测
        try:
            board_state = self.encoder.encode_board(board, self.color)
            policy_probs, value = self.model.predict(board_state)
            
            # 只考虑合法移动
            move_probs = []
            for move in legal_moves:
                move_idx = self.encoder.encode_move(move[0], move[1])
                prob = policy_probs[move_idx]
                move_probs.append((move, prob))
            
            # 根据温度参数选择移动
            if self.temperature < 0.01:
                # 贪心选择
                best_move = max(move_probs, key=lambda x: x[1])[0]
            else:
                # 按概率采样
                moves = [m for m, p in move_probs]
                probs = np.array([p for m, p in move_probs])
                
                # 温度缩放
                probs = probs ** (1.0 / self.temperature)
                probs = probs / probs.sum()
                
                # 采样
                idx = np.random.choice(len(moves), p=probs)
                best_move = moves[idx]
            
            return best_move
            
        except Exception as e:
            print(f"神经网络预测失败: {e}")
            # 降级为随机移动
            import random
            return random.choice(legal_moves)
    
    def set_temperature(self, temperature):
        """设置温度参数"""
        self.temperature = temperature
    
    def evaluate_position(self, board, game_state):
        """评估当前局面的价值"""
        try:
            board_state = self.encoder.encode_board(board, self.color)
            _, value = self.model.predict(board_state)
            return value
        except:
            return 0.0
