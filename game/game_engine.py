from typing import Optional, Tuple
from core.interfaces import Observable
from core.game_state import GameState
from game.dark_chess_board import DarkChessBoard

class GameEngine(Observable):
    """游戏引擎 - 分离逻辑与视图"""
    def __init__(self, red_player, black_player):
        self.board = DarkChessBoard()
        self.red_player = red_player
        self.black_player = black_player
        self.game_state = GameState()
        self.observers = []
        
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
        
        if not self.game_state.game_over:
            self.game_state.switch_turn()
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