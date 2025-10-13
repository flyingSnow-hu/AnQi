from enum import Enum
from typing import List, Tuple, Set
from core.interfaces import PieceState

class PieceType(Enum):
    GENERAL = "general"    # 将/帅
    ADVISOR = "advisor"    # 士/仕
    ELEPHANT = "elephant"  # 象/相
    HORSE = "horse"        # 马
    CHARIOT = "chariot"    # 车
    CANNON = "cannon"      # 炮
    SOLDIER = "soldier"    # 兵/卒

class DarkChessPiece:
    """揭棋棋子类"""
    def __init__(self, piece_type: PieceType, color: str):
        self.piece_type = piece_type
        self.color = color  # "red" or "black"
        self.state = PieceState.HIDDEN if piece_type != PieceType.GENERAL else PieceState.REVEALED
        self.position = None
        
    @property
    def symbol(self) -> str:
        """获取棋子显示符号"""
        if self.state == PieceState.HIDDEN:
            return "●" if self.color == "black" else "○"  # 背面显示
        
        # 根据颜色显示不同的名称
        if self.piece_type == PieceType.GENERAL:
            return "帅" if self.color == "red" else "将"
        elif self.piece_type == PieceType.ADVISOR:
            return "仕" if self.color == "red" else "士"
        elif self.piece_type == PieceType.ELEPHANT:
            return "相" if self.color == "red" else "象"
        elif self.piece_type == PieceType.HORSE:
            return "马"
        elif self.piece_type == PieceType.CHARIOT:
            return "车"
        elif self.piece_type == PieceType.CANNON:
            return "炮"
        elif self.piece_type == PieceType.SOLDIER:
            return "兵" if self.color == "red" else "卒"
        return "?"
        
    def reveal(self):
        """翻开棋子"""
        self.state = PieceState.REVEALED
        
    def is_revealed(self) -> bool:
        return self.state == PieceState.REVEALED
        
    def can_capture(self, target_piece) -> bool:
        """判断是否可以吃掉目标棋子（只要颜色不同即可，不需要判断是否翻开）"""
        if self.color == target_piece.color:
            return False
        # 可以吃对方任何棋子，无论是否翻开
        return True
    
    def _get_piece_type_by_position(self, row: int, col: int) -> PieceType:
        """根据位置推断棋子类型（用于未翻开的棋子）"""
        # 红方位置判断
        if self.color == "red":
            if row == 9:  # 第一排
                if col in [0, 8]:
                    return PieceType.CHARIOT  # 车
                elif col in [1, 7]:
                    return PieceType.HORSE    # 马
                elif col in [2, 6]:
                    return PieceType.ELEPHANT # 相
                elif col in [3, 5]:
                    return PieceType.ADVISOR  # 仕
                elif col == 4:
                    return PieceType.GENERAL  # 帅
            elif row == 7 and col in [1, 7]:
                return PieceType.CANNON       # 炮
            elif row == 6 and col in [0, 2, 4, 6, 8]:
                return PieceType.SOLDIER      # 兵
        
        # 黑方位置判断
        else:  # black
            if row == 0:  # 第一排
                if col in [0, 8]:
                    return PieceType.CHARIOT  # 车
                elif col in [1, 7]:
                    return PieceType.HORSE    # 马
                elif col in [2, 6]:
                    return PieceType.ELEPHANT # 象
                elif col in [3, 5]:
                    return PieceType.ADVISOR  # 士
                elif col == 4:
                    return PieceType.GENERAL  # 将
            elif row == 2 and col in [1, 7]:
                return PieceType.CANNON       # 炮
            elif row == 3 and col in [0, 2, 4, 6, 8]:
                return PieceType.SOLDIER      # 卒
        
        # 如果不在预期位置，返回兵/卒（最安全的默认值）
        return PieceType.SOLDIER
        
    def get_valid_moves(self, board, from_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """获取有效移动位置"""
        row, col = from_pos
        
        if not self.is_revealed():
            # 未翻开的棋子：根据位置推断兵种，按该兵种走法
            inferred_type = self._get_piece_type_by_position(row, col)
            return self._get_moves_by_type(board, row, col, inferred_type, is_revealed=False)
        else:
            # 已翻开按真实兵种走法
            return self._get_moves_by_type(board, row, col, self.piece_type, is_revealed=True)
                
    def _get_all_possible_moves(self, board, row: int, col: int) -> List[Tuple[int, int]]:
        """未翻开棋子的所有可能走法（所有兵种走法的并集）"""
        # 这个方法现在不再使用，但保留以防需要
        all_moves = set()
        
        # 因为棋子是随机摆放的，可能是任何兵种（除了将帅）
        possible_types = {
            PieceType.ADVISOR, PieceType.ELEPHANT, PieceType.HORSE, 
            PieceType.CHARIOT, PieceType.CANNON, PieceType.SOLDIER
        }
        
        for piece_type in possible_types:
            moves = self._get_moves_by_type(board, row, col, piece_type)
            all_moves.update(moves)
            
        return list(all_moves)
        
    def _get_moves_by_type(self, board, row: int, col: int, piece_type: PieceType, is_revealed: bool = True) -> List[Tuple[int, int]]:
        """根据指定兵种获取走法"""
        if piece_type == PieceType.GENERAL:
            return self._get_general_moves(board, row, col)
        elif piece_type == PieceType.ADVISOR:
            return self._get_advisor_moves(board, row, col, is_revealed)
        elif piece_type == PieceType.ELEPHANT:
            return self._get_elephant_moves(board, row, col)
        elif piece_type == PieceType.HORSE:
            return self._get_horse_moves(board, row, col)
        elif piece_type == PieceType.CHARIOT:
            return self._get_chariot_moves(board, row, col)
        elif piece_type == PieceType.CANNON:
            return self._get_cannon_moves(board, row, col)
        elif piece_type == PieceType.SOLDIER:
            return self._get_soldier_moves(board, row, col)
        return []
        
    def _get_general_moves(self, board, row: int, col: int) -> List[Tuple[int, int]]:
        """将的移动：只能在九宫格内走一格，可以吃对方任何棋子。飞将规则：如果与对方将/帅在同一列且中间无遮挡，可以直接飞过去吃掉"""
        valid_moves = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        # 普通移动：九宫格内走一格
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if self._is_in_palace(new_row, new_col) and board.is_valid_position(new_row, new_col):
                target = board.get_piece(new_row, new_col)
                if not target:
                    valid_moves.append((new_row, new_col))
                elif target.color != self.color:
                    # 可以吃对方任何棋子（无论是否翻开）
                    valid_moves.append((new_row, new_col))
        
        # 飞将规则：检查同一列是否有对方的将/帅，且中间无遮挡
        for check_row in range(10):
            if check_row == row:
                continue
            target = board.get_piece(check_row, col)
            if target and target.color != self.color and target.piece_type == PieceType.GENERAL:
                # 找到对方的将/帅，检查中间是否有遮挡
                min_row = min(row, check_row)
                max_row = max(row, check_row)
                has_blocking = False
                for middle_row in range(min_row + 1, max_row):
                    if board.get_piece(middle_row, col):
                        has_blocking = True
                        break
                
                # 如果中间没有遮挡，可以飞过去吃掉对方将/帅
                if not has_blocking:
                    valid_moves.append((check_row, col))
                break
        
        return valid_moves
        
    def _get_advisor_moves(self, board, row: int, col: int, is_revealed: bool = True) -> List[Tuple[int, int]]:
        """士的移动：未翻开时只能在九宫格内斜走一格，翻开后可以任意斜走，可以吃对方任何棋子"""
        valid_moves = []
        directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if board.is_valid_position(new_row, new_col):
                # 未翻开的士必须在九宫格内
                if not is_revealed and not self._is_in_palace(new_row, new_col):
                    continue
                target = board.get_piece(new_row, new_col)
                if not target:
                    valid_moves.append((new_row, new_col))
                elif target.color != self.color:
                    # 可以吃对方任何棋子（无论是否翻开）
                    valid_moves.append((new_row, new_col))
        return valid_moves
        
    def _get_elephant_moves(self, board, row: int, col: int) -> List[Tuple[int, int]]:
        """象的移动：走田字，可以过河，可以吃对方任何棋子"""
        valid_moves = []
        directions = [(2, 2), (2, -2), (-2, 2), (-2, -2)]
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if board.is_valid_position(new_row, new_col):
                # 检查象眼
                eye_row, eye_col = row + dr//2, col + dc//2
                if not board.get_piece(eye_row, eye_col):
                    target = board.get_piece(new_row, new_col)
                    if not target:
                        valid_moves.append((new_row, new_col))
                    elif target.color != self.color:
                        # 可以吃对方任何棋子（无论是否翻开）
                        valid_moves.append((new_row, new_col))
        return valid_moves
        
    def _get_horse_moves(self, board, row: int, col: int) -> List[Tuple[int, int]]:
        """马的移动：走日字，可以吃对方任何棋子"""
        valid_moves = []
        horse_moves = [(2, 1), (2, -1), (-2, 1), (-2, -1), 
                      (1, 2), (1, -2), (-1, 2), (-1, -2)]
        
        for dr, dc in horse_moves:
            new_row, new_col = row + dr, col + dc
            if board.is_valid_position(new_row, new_col):
                # 检查马腿
                if not self._horse_leg_blocked(board, row, col, dr, dc):
                    target = board.get_piece(new_row, new_col)
                    if not target:
                        valid_moves.append((new_row, new_col))
                    elif target.color != self.color:
                        # 可以吃对方任何棋子（无论是否翻开）
                        valid_moves.append((new_row, new_col))
        return valid_moves
        
    def _get_chariot_moves(self, board, row: int, col: int) -> List[Tuple[int, int]]:
        """车的移动：直线走，可以吃对方任何棋子"""
        valid_moves = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for dr, dc in directions:
            for i in range(1, 10):
                new_row, new_col = row + dr * i, col + dc * i
                if not board.is_valid_position(new_row, new_col):
                    break
                target = board.get_piece(new_row, new_col)
                if target:
                    if target.color != self.color:
                        # 可以吃对方任何棋子（无论是否翻开）
                        valid_moves.append((new_row, new_col))
                    break
                else:
                    valid_moves.append((new_row, new_col))
        return valid_moves
        
    def _get_cannon_moves(self, board, row: int, col: int) -> List[Tuple[int, int]]:
        """炮的移动：不吃子时直线走，吃子时需要跳一个棋子（不论跳板是否翻开，可以吃对方任何棋子）"""
        valid_moves = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for dr, dc in directions:
            jumped = False
            for i in range(1, 10):
                new_row, new_col = row + dr * i, col + dc * i
                if not board.is_valid_position(new_row, new_col):
                    break
                    
                target = board.get_piece(new_row, new_col)
                if target:
                    if not jumped:
                        # 第一个棋子作为跳板（不论是否翻开，不论是敌是友）
                        jumped = True
                    else:
                        # 遇到第二个棋子，如果是对方棋子（无论是否翻开）则可以吃
                        if target.color != self.color:
                            valid_moves.append((new_row, new_col))
                        # 无论能否吃，都要停止（不能继续往前跳）
                        break
                else:
                    # 空位置
                    if not jumped:
                        # 还没跳过棋子，可以移动到空位
                        valid_moves.append((new_row, new_col))
                    # 如果已经跳过一个棋子，不能移动到空位（必须吃子）
        return valid_moves
        
    def _get_soldier_moves(self, board, row: int, col: int) -> List[Tuple[int, int]]:
        """兵的移动：向前一格，过河后可横走，可以吃对方任何棋子"""
        valid_moves = []
        
        if self.color == "red":
            directions = [(-1, 0)]  # 红方向上
            if row < 5:  # 过河后可横走
                directions.extend([(0, -1), (0, 1)])
        else:
            directions = [(1, 0)]   # 黑方向下
            if row > 4:  # 过河后可横走
                directions.extend([(0, -1), (0, 1)])
                
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if board.is_valid_position(new_row, new_col):
                target = board.get_piece(new_row, new_col)
                if not target:
                    valid_moves.append((new_row, new_col))
                elif target.color != self.color:
                    # 可以吃对方任何棋子（无论是否翻开）
                    valid_moves.append((new_row, new_col))
        return valid_moves
        
    def _is_in_palace(self, row: int, col: int) -> bool:
        """检查是否在九宫格内"""
        if self.color == "red":
            return 7 <= row <= 9 and 3 <= col <= 5
        else:
            return 0 <= row <= 2 and 3 <= col <= 5
            
    def _horse_leg_blocked(self, board, row: int, col: int, dr: int, dc: int) -> bool:
        """检查马腿是否被挡"""
        if abs(dr) == 2:
            leg_row = row + dr // 2
            return board.get_piece(leg_row, col) is not None
        else:
            leg_col = col + dc // 2
            return board.get_piece(row, leg_col) is not None