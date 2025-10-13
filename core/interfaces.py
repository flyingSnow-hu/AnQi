from abc import ABC, abstractmethod
from typing import Tuple, Optional
from enum import Enum

class PieceState(Enum):
    """棋子状态"""
    HIDDEN = "hidden"      # 未翻开（背面朝上）
    REVEALED = "revealed"  # 已翻开

# 基础接口定义
class Playable(ABC):
    """可游戏接口"""
    @abstractmethod
    def get_move(self, board, game_state) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        获取玩家的移动
        返回: (from_pos, to_pos) 或 None
        """
        pass

class Observable(ABC):
    """观察者模式接口"""
    @abstractmethod
    def add_observer(self, observer) -> None:
        pass
    
    @abstractmethod
    def remove_observer(self, observer) -> None:
        pass
    
    @abstractmethod
    def notify_observers(self, event: str, data) -> None:
        pass