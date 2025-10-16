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
        
        # 长将检测
        self.consecutive_checks = 0  # 连续将军次数
        self.checking_side = None    # 将军方('red'或'black')
        self.is_loss_by_perpetual_check = False  # 是否因长将判负
        self.perpetual_check_threshold = 6  # 长将判负阈值(连续6次)
        
        # 长捉检测
        self.consecutive_chases = 0  # 连续追捉次数
        self.chasing_side = None     # 追捉方('red'或'black')
        self.chased_piece_pos = None # 被追捉棋子的位置
        self.is_loss_by_perpetual_chase = False  # 是否因长捉判负
        self.perpetual_chase_threshold = 6  # 长捉判负阈值(连续6次)
        
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
    
    def update_check_status(self, is_checking, checking_side):
        """
        更新将军状态
        
        Args:
            is_checking: 当前是否有将军
            checking_side: 如果有将军,是哪一方在将军
        """
        if is_checking:
            if self.checking_side == checking_side:
                # 同一方连续将军
                self.consecutive_checks += 1
            else:
                # 新的将军方,重置计数
                self.checking_side = checking_side
                self.consecutive_checks = 1
            
            # 检查是否达到长将阈值
            if self.consecutive_checks >= self.perpetual_check_threshold:
                self.is_loss_by_perpetual_check = True
                self.game_over = True
                # 长将方判负,对方获胜
                opponent = "black" if checking_side == "red" else "red"
                self.winner = "黑方" if opponent == "black" else "红方"
                return True
        else:
            # 没有将军,重置计数
            self.consecutive_checks = 0
            self.checking_side = None
        
        return False
    
    def get_perpetual_check_penalty(self):
        """
        获取长将惩罚
        用于训练时惩罚长将行为
        
        Returns:
            惩罚值(负数),连续将军次数越多惩罚越大
        """
        if self.consecutive_checks == 0:
            return 0.0
        elif self.consecutive_checks < 3:
            return -0.005  # 轻微惩罚
        elif self.consecutive_checks < 5:
            return -0.02   # 中等惩罚
        else:
            return -0.1    # 严重惩罚(接近长将判负)
    
    def update_chase_status(self, is_chasing, chasing_side, chased_piece_pos):
        """
        更新追捉状态
        
        Args:
            is_chasing: 当前是否在追捉
            chasing_side: 如果在追捉,是哪一方在追捉
            chased_piece_pos: 被追捉棋子的位置
        """
        if is_chasing:
            # 检查是否在追捉同一个棋子
            if (self.chasing_side == chasing_side and 
                self.chased_piece_pos == chased_piece_pos):
                # 同一方追捉同一个棋子
                self.consecutive_chases += 1
            else:
                # 新的追捉目标,重置计数
                self.chasing_side = chasing_side
                self.chased_piece_pos = chased_piece_pos
                self.consecutive_chases = 1
            
            # 检查是否达到长捉阈值
            if self.consecutive_chases >= self.perpetual_chase_threshold:
                self.is_loss_by_perpetual_chase = True
                self.game_over = True
                # 长捉方判负,对方获胜
                opponent = "black" if chasing_side == "red" else "red"
                self.winner = "黑方" if opponent == "black" else "红方"
                return True
        else:
            # 没有追捉,重置计数
            self.consecutive_chases = 0
            self.chasing_side = None
            self.chased_piece_pos = None
        
        return False
    
    def get_perpetual_chase_penalty(self):
        """
        获取长捉惩罚
        用于训练时惩罚长捉行为
        
        Returns:
            惩罚值(负数),连续追捉次数越多惩罚越大
        """
        if self.consecutive_chases == 0:
            return 0.0
        elif self.consecutive_chases < 3:
            return -0.003  # 轻微惩罚
        elif self.consecutive_chases < 5:
            return -0.015  # 中等惩罚
        else:
            return -0.08   # 严重惩罚(接近长捉判负)