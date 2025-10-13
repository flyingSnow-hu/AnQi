from abc import ABC, abstractmethod
from typing import Optional, Tuple
from core.interfaces import Playable

class BasePlayer(Playable, ABC):
    """玩家基类"""
    def __init__(self, name: str, color: str = ""):
        self.name = name
        self.color = color
        
    @abstractmethod
    def get_move(self, board, game_state) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        pass
    
    @property
    @abstractmethod
    def player_type(self) -> str:
        pass