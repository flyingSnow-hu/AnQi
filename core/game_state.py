from typing import Dict, List, Any

class GameState:
    """揭棋游戏状态管理"""
    def __init__(self):
        self.current_turn = "red"  # 红方先行
        self.move_count = 0
        self.game_over = False
        self.winner = None
        self.last_move = None
        self.captured_pieces = {"red": [], "black": []}
        self.move_history = []
        
    def switch_turn(self):
        """切换回合"""
        self.current_turn = "black" if self.current_turn == "red" else "red"
        self.move_count += 1
        
    def add_move_to_history(self, move_info: Dict[str, Any]):
        """添加移动记录到历史"""
        self.move_history.append(move_info)
        self.last_move = move_info
        
    def end_game(self, winner: str):
        """结束游戏"""
        self.game_over = True
        self.winner = winner