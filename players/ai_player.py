from .base_player import BasePlayer
from typing import Optional, Tuple

class AIPlayer(BasePlayer):
    """AI玩家 - 预留接口"""
    def __init__(self, name: str, color: str = "", difficulty: int = 1):
        super().__init__(name, color)
        self.difficulty = difficulty
        
    def get_move(self, board, game_state) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        # TODO: 实现AI算法（极大极小值+剪枝）
        return None
        
    @property
    def player_type(self) -> str:
        return "ai"