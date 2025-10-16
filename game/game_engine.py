from typing import Optional, Tuple
from core.interfaces import Observable
from core.game_state import GameState
from game.dark_chess_board import DarkChessBoard
from game.zobrist_hash import ZobristHash

class GameEngine(Observable):
    """游戏引擎 - 分离逻辑与视图"""
    def __init__(self, red_player, black_player):
        self.board = DarkChessBoard()
        self.red_player = red_player
        self.black_player = black_player
        self.game_state = GameState()
        self.observers = []
        
        # Zobrist哈希用于重复局面检测
        self.zobrist = ZobristHash()
        
        # 记录初始局面哈希
        initial_hash = self.zobrist.compute_hash(self.board, self.game_state.current_turn)
        self.game_state.add_position_hash(initial_hash)
        
        # 设置玩家颜色
        self.red_player.color = "red"
        self.black_player.color = "black"
        
        # AI玩家支持
        self.ai_players = {'red': None, 'black': None}
        
    def add_observer(self, observer):
        self.observers.append(observer)
        
    def remove_observer(self, observer):
        if observer in self.observers:
            self.observers.remove(observer)
            
    def notify_observers(self, event: str, data):
        for observer in self.observers:
            observer.on_game_event(event, data)
            
    @property
    def current_player(self):
        return self.red_player if self.game_state.current_turn == "red" else self.black_player
        
    def make_move(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> bool:
        """执行移动"""
        if self.game_state.game_over:
            return False
            
        if not self._is_valid_move(from_pos, to_pos):
            return False
            
        # 记录移动前的状态
        piece = self.board.get_piece(*from_pos)
        was_hidden = not piece.is_revealed() if piece else False
        
        # 执行移动
        captured_piece = self.board.move_piece(from_pos, to_pos)
        
        # 记录移动历史
        move_record = {
            "from": from_pos,
            "to": to_pos,
            "piece": piece.piece_type.value if piece else None,
            "piece_color": piece.color if piece else None,
            "captured": captured_piece.piece_type.value if captured_piece else None,
            "captured_color": captured_piece.color if captured_piece else None,
            "was_hidden": was_hidden,
            "turn": self.game_state.current_turn,
            "move_number": self.game_state.move_count + 1
        }
        
        self.game_state.add_move_to_history(move_record)
        
        # 通知观察者移动完成
        self.notify_observers("move_made", move_record)
        
        # 如果有棋子被吃掉，添加到被吃列表
        if captured_piece:
            self.game_state.captured_pieces[captured_piece.color].append(captured_piece)
        
        # 检查游戏状态
        self._update_game_status()
        
        # 计算移动后的局面哈希并检查重复
        if not self.game_state.game_over:
            # 先切换回合,再计算哈希(因为哈希包含行棋方信息)
            self.game_state.switch_turn()
            
            # 检测长将
            is_perpetual_check = self.check_perpetual_check()
            
            # 检测长捉(只在没有长将时检测)
            is_perpetual_chase = False
            if not is_perpetual_check:
                is_perpetual_chase = self.check_perpetual_chase()
            
            if not is_perpetual_check and not is_perpetual_chase:
                # 计算新局面哈希
                current_hash = self.zobrist.compute_hash(self.board, self.game_state.current_turn)
                is_draw = self.game_state.add_position_hash(current_hash)
                
                if is_draw:
                    # 三次重复判和
                    self.notify_observers("game_over", {
                        "winner": None,
                        "reason": "三次重复局面判和"
                    })
                else:
                    self.notify_observers("turn_changed", {
                        "current_player": self.current_player.name,
                        "color": self.game_state.current_turn
                    })
        
        return True
        
    def _is_valid_move(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> bool:
        """验证移动合法性"""
        piece = self.board.get_piece(*from_pos)
        
        if not piece:
            return False
            
        if piece.color != self.game_state.current_turn:
            return False
            
        return self.board.is_valid_move(from_pos, to_pos, self.game_state.current_turn)
        
    def _update_game_status(self):
        """更新游戏状态"""
        # 检查将死 - 如果某方的将被吃掉，游戏结束
        if self.board.is_general_captured("red"):
            self.game_state.end_game("黑方")
            self.notify_observers("game_over", {
                "winner": "黑方",
                "reason": "红方将帅被吃"
            })
        elif self.board.is_general_captured("black"):
            self.game_state.end_game("红方")
            self.notify_observers("game_over", {
                "winner": "红方", 
                "reason": "黑方将帅被吃"
            })
    
    def is_in_check(self, defending_color: str) -> bool:
        """
        检查某方是否被将军
        
        Args:
            defending_color: 被检查的一方('red'或'black')
        
        Returns:
            True表示被将军，False表示未被将军
        """
        # 找到defending_color的将/帅位置
        general_pos = self._find_general(defending_color)
        if general_pos is None:
            return False  # 将帅已被吃，按理不应该到这里
        
        # 检查attacking_color是否有棋子能吃掉将/帅
        attacking_color = "black" if defending_color == "red" else "red"
        
        # 遍历所有attacking_color的棋子
        for row in range(10):
            for col in range(9):
                piece = self.board.get_piece(row, col)
                if piece and piece.color == attacking_color and piece.is_revealed():
                    # 检查该棋子是否能移动到将帅位置
                    if self.board.is_valid_move((row, col), general_pos, attacking_color):
                        return True
        
        return False
    
    def _find_general(self, color: str) -> Optional[Tuple[int, int]]:
        """
        查找某方将/帅的位置
        
        Args:
            color: 'red'或'black'
        
        Returns:
            将帅位置(row, col)，如果未找到返回None
        """
        from game.dark_chess_piece import PieceType
        
        for row in range(10):
            for col in range(9):
                piece = self.board.get_piece(row, col)
                if piece and piece.color == color and piece.piece_type == PieceType.GENERAL:
                    return (row, col)
        
        return None
    
    def has_non_check_response(self, defending_color: str) -> bool:
        """
        检查某方是否有非应将的合法走法
        
        Args:
            defending_color: 被将军的一方
        
        Returns:
            True表示有非应将走法，False表示只能应将
        """
        # 获取所有defending_color的棋子
        for row in range(10):
            for col in range(9):
                piece = self.board.get_piece(row, col)
                if piece and piece.color == defending_color:
                    # 获取该棋子的所有合法移动
                    valid_moves = piece.get_valid_moves(self.board, (row, col))
                    
                    for to_pos in valid_moves:
                        # 模拟走这步棋
                        original_piece = self.board.get_piece(*to_pos)
                        self.board.set_piece(to_pos[0], to_pos[1], piece)
                        self.board.set_piece(row, col, None)
                        
                        # 检查走完后是否还被将军
                        still_in_check = self.is_in_check(defending_color)
                        
                        # 撤销移动
                        self.board.set_piece(row, col, piece)
                        self.board.set_piece(to_pos[0], to_pos[1], original_piece)
                        
                        # 如果走完后不再被将军，说明有非应将走法
                        if not still_in_check:
                            return True
        
        return False  # 所有走法都无法解除将军
    
    def is_piece_under_threat(self, piece_pos: Tuple[int, int], by_color: str) -> bool:
        """
        检查某个位置的棋子是否被某方威胁(可以被吃)
        
        Args:
            piece_pos: 被检查棋子的位置
            by_color: 威胁方的颜色
        
        Returns:
            True表示被威胁，False表示未被威胁
        """
        target_piece = self.board.get_piece(*piece_pos)
        if not target_piece:
            return False
        
        # 遍历by_color的所有棋子
        for row in range(10):
            for col in range(9):
                piece = self.board.get_piece(row, col)
                if piece and piece.color == by_color and piece.is_revealed():
                    # 检查该棋子是否能移动到目标位置
                    if self.board.is_valid_move((row, col), piece_pos, by_color):
                        return True
        
        return False
    
    def can_piece_escape(self, piece_pos: Tuple[int, int], from_color: str) -> bool:
        """
        检查某个位置的棋子是否能逃脱威胁
        
        Args:
            piece_pos: 被追捉棋子的位置
            from_color: 威胁方的颜色
        
        Returns:
            True表示能逃脱(有安全走法)，False表示只能被动逃跑
        """
        piece = self.board.get_piece(*piece_pos)
        if not piece:
            return True  # 棋子不存在,算能逃脱
        
        # 获取该棋子的所有合法移动
        valid_moves = piece.get_valid_moves(self.board, piece_pos)
        
        for to_pos in valid_moves:
            # 模拟走这步棋
            original_piece = self.board.get_piece(*to_pos)
            self.board.set_piece(to_pos[0], to_pos[1], piece)
            self.board.set_piece(piece_pos[0], piece_pos[1], None)
            
            # 检查走完后是否还被威胁
            still_under_threat = self.is_piece_under_threat(to_pos, from_color)
            
            # 撤销移动
            self.board.set_piece(piece_pos[0], piece_pos[1], piece)
            self.board.set_piece(to_pos[0], to_pos[1], original_piece)
            
            # 如果走完后不再被威胁,说明能逃脱
            if not still_under_threat:
                return True
        
        return False  # 所有走法都无法逃脱威胁
    
    def find_chased_piece(self, attacking_color: str) -> Optional[Tuple[int, int]]:
        """
        查找被追捉的棋子
        
        Args:
            attacking_color: 攻击方的颜色
        
        Returns:
            被追捉棋子的位置，如果没有则返回None
        """
        defending_color = "black" if attacking_color == "red" else "red"
        
        # 找出所有被威胁的对方棋子
        threatened_pieces = []
        for row in range(10):
            for col in range(9):
                piece = self.board.get_piece(row, col)
                if piece and piece.color == defending_color and piece.is_revealed():
                    # 排除将帅(将帅的威胁属于"将军",不属于"捉")
                    from game.dark_chess_piece import PieceType
                    if piece.piece_type != PieceType.GENERAL:
                        if self.is_piece_under_threat((row, col), attacking_color):
                            # 检查该棋子是否只能被动逃跑
                            if not self.can_piece_escape((row, col), attacking_color):
                                threatened_pieces.append((row, col))
        
        # 如果只有一个被追捉的棋子,返回它
        # 如果有多个,返回第一个(简化处理)
        if threatened_pieces:
            return threatened_pieces[0]
        
        return None
    
    def check_perpetual_chase(self) -> bool:
        """
        检测长捉
        
        Returns:
            True表示触发长捉判负，False表示未触发
        """
        # 当前是谁刚走完棋(可能是追捉方)
        last_moved_color = "black" if self.game_state.current_turn == "red" else "red"
        
        # 查找是否有被追捉的棋子
        chased_piece_pos = self.find_chased_piece(last_moved_color)
        
        if chased_piece_pos:
            # 有被追捉的棋子,更新追捉状态
            is_perpetual = self.game_state.update_chase_status(
                True, last_moved_color, chased_piece_pos
            )
            
            if is_perpetual:
                # 触发长捉判负
                self.notify_observers("game_over", {
                    "winner": self.game_state.winner,
                    "reason": f"长捉判负(连续追捉{self.game_state.consecutive_chases}次)"
                })
                return True
        else:
            # 没有追捉,重置计数
            self.game_state.update_chase_status(False, None, None)
        
        return False
    
    def check_perpetual_check(self) -> bool:
        """
        检测长将
        
        Returns:
            True表示触发长将判负，False表示未触发
        """
        # 当前是谁刚走完棋(可能是将军方)
        last_moved_color = "black" if self.game_state.current_turn == "red" else "red"
        opponent_color = self.game_state.current_turn
        
        # 检查对方是否被将军
        if self.is_in_check(opponent_color):
            # 检查对方是否只能应将(没有非应将走法)
            has_other_moves = self.has_non_check_response(opponent_color)
            
            if not has_other_moves:
                # 对方被迫应将，更新将军状态
                is_perpetual = self.game_state.update_check_status(True, last_moved_color)
                
                if is_perpetual:
                    # 触发长将判负
                    self.notify_observers("game_over", {
                        "winner": self.game_state.winner,
                        "reason": f"长将判负(连续{self.game_state.consecutive_checks}次)"
                    })
                    return True
            else:
                # 对方有其他走法，重置将军计数
                self.game_state.update_check_status(False, None)
        else:
            # 没有将军，重置计数
            self.game_state.update_check_status(False, None)
        
        return False
            
    def get_valid_moves_for_position(self, pos: Tuple[int, int]) -> list:
        """获取指定位置棋子的有效移动"""
        piece = self.board.get_piece(*pos)
        if piece and piece.color == self.game_state.current_turn:
            return piece.get_valid_moves(self.board, pos)
        return []
        
    def set_ai_player(self, color: str, ai_player):
        """设置AI玩家"""
        self.ai_players[color] = ai_player
        
    def get_ai_move(self) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """获取当前AI的移动"""
        current_color = self.game_state.current_turn
        ai = self.ai_players.get(current_color)
        if ai:
            return ai.get_move(self.board, self.game_state)
        return None
    
    def is_ai_turn(self) -> bool:
        """判断当前是否是AI回合"""
        return self.ai_players.get(self.game_state.current_turn) is not None