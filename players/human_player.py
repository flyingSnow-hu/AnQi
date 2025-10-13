from .base_player import BasePlayer
from typing import Optional, Tuple

class HumanPlayer(BasePlayer):
    """人类玩家"""
    def __init__(self, name: str, color: str = ""):
        super().__init__(name, color)
        
    def get_move(self, board, game_state) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        # 人类玩家通过GUI设置移动
        return None
        
    @property
    def player_type(self) -> str:
        return "human"