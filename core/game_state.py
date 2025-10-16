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
        
        # 重复局面检测
        self.position_history = []  # 记录历史局面哈希
        self.repeat_count = {}      # 统计每个局面出现次数
        self.is_draw_by_repetition = False  # 是否因重复判和
        self.repetition_penalty = 0.0  # 重复局面惩罚值
        
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
    
    def add_position_hash(self, position_hash: int):
        """
        添加局面哈希并检测重复
        
        Args:
            position_hash: 当前局面的哈希值
        
        Returns:
            是否触发三次重复
        """
        self.position_history.append(position_hash)
        
        # 更新重复计数
        if position_hash not in self.repeat_count:
            self.repeat_count[position_hash] = 0
        self.repeat_count[position_hash] += 1
        
        # 检查三次重复
        if self.repeat_count[position_hash] >= 3:
            self.is_draw_by_repetition = True
            self.game_over = True
            self.winner = None  # 平局
            return True
        
        # 计算重复惩罚
        repeat_times = self.repeat_count[position_hash]
        if repeat_times == 1:
            self.repetition_penalty = 0.0
        elif repeat_times == 2:
            self.repetition_penalty = -0.01  # 第二次重复,小惩罚
        else:  # >= 3
            self.repetition_penalty = -0.05  # 第三次重复(判和),较大惩罚
        
        return False
    
    def get_repetition_penalty(self):
        """获取当前的重复局面惩罚值"""
        return self.repetition_penalty
    
    def reset_repetition_tracking(self):
        """重置重复局面追踪"""
        self.position_history = []
        self.repeat_count = {}
        self.is_draw_by_repetition = False
        self.repetition_penalty = 0.0