"""
AI玩家基类
处理AI的信息可见性和基础功能
"""
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Dict
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.dark_chess_piece import PieceType

class AIPlayer(ABC):
    """AI玩家抽象基类"""
    
    def __init__(self, color: str):
        """
        初始化AI玩家
        color: "red" or "black"
        """
        self.name = f"AI_{color}"
        self.color = color
        self.opponent_color = "black" if color == "red" else "red"
        
    @abstractmethod
    def get_move(self, board, game_state) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        获取AI的移动决策
        返回: (from_pos, to_pos)
        """
        pass
    
    def get_visible_board_state(self, board, game_state) -> Dict:
        """
        获取AI可见的棋盘状态
        AI只能看到：
        1. 所有已翻开的棋子
        2. 自己吃掉的暗棋
        3. 对方吃掉的暗棋对AI不可见（只知道被吃了，但不知道是什么）
        """
        visible_info = {
            'revealed_pieces': [],  # 所有已翻开的棋子
            'hidden_pieces': [],    # 未翻开棋子的位置
            'known_captured': [],   # 已知的被吃棋子（自己吃的暗棋）
            'unknown_captured_count': {'red': 0, 'black': 0}  # 未知的被吃棋子数量
        }
        
        # 收集棋盘上的棋子信息
        for row in range(10):
            for col in range(9):
                piece = board.get_piece(row, col)
                if piece:
                    if piece.is_revealed():
                        visible_info['revealed_pieces'].append({
                            'pos': (row, col),
                            'type': piece.piece_type,
                            'color': piece.color
                        })
                    else:
                        visible_info['hidden_pieces'].append({
                            'pos': (row, col),
                            'color': piece.color  # AI能看到暗棋的颜色（红○/黑●）
                        })
        
        # 收集被吃棋子信息
        if hasattr(game_state, 'captured_visibility'):
            for color in ['red', 'black']:
                visibility_list = game_state.captured_visibility[color]
                for piece, visible_to in visibility_list:
                    if visible_to == "both" or visible_to == self.color:
                        # AI能看到的被吃棋子
                        visible_info['known_captured'].append({
                            'type': piece.piece_type,
                            'color': piece.color,
                            'was_hidden': visible_to == self.color and visible_to != "both"
                        })
                    else:
                        # AI不知道具体是什么的被吃棋子
                        visible_info['unknown_captured_count'][color] += 1
        
        return visible_info
    
    def get_all_legal_moves(self, board, color: str) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """获取指定颜色所有合法移动（包括未翻开的棋子）"""
        legal_moves = []
        
        for row in range(10):
            for col in range(9):
                piece = board.get_piece(row, col)
                # 修改：移除 is_revealed() 限制，AI可以移动所有己方棋子
                if piece and piece.color == color:
                    valid_moves = piece.get_valid_moves(board, (row, col))
                    for to_pos in valid_moves:
                        legal_moves.append(((row, col), to_pos))
        
        return legal_moves
