import random
from typing import List, Tuple, Optional
from .dark_chess_piece import DarkChessPiece, PieceType, PieceState

class DarkChessBoard:
    """揭棋棋盘"""
    def __init__(self):
        self.board = [[None for _ in range(9)] for _ in range(10)]  # 10x9标准象棋棋盘
        self.setup_board()
        
    def setup_board(self):
        """初始化棋盘 - 将帅保持原位并翻开，红方和黑方棋子都按传统象棋位置摆放但随机交换，所有非将帅棋子背面朝上"""
        # 清空棋盘
        for row in range(10):
            for col in range(9):
                self.board[row][col] = None
                
        # 首先放置将帅在固定位置（已翻开）
        black_general = DarkChessPiece(PieceType.GENERAL, "black")
        black_general.state = PieceState.REVEALED
        self.set_piece(0, 4, black_general)
        
        red_general = DarkChessPiece(PieceType.GENERAL, "red")
        red_general.state = PieceState.REVEALED
        self.set_piece(9, 4, red_general)
        
        # 红方棋子按照传统象棋位置摆放，但除了帅之外随机交换位置（背面朝上）
        # 创建红方所有棋子（除将帅）
        red_pieces_list = []
        
        # 第一排（row=9）：车马相士帅士相马车（除了帅）
        red_pieces_list.append(DarkChessPiece(PieceType.CHARIOT, "red"))   # 左车
        red_pieces_list.append(DarkChessPiece(PieceType.HORSE, "red"))     # 左马
        red_pieces_list.append(DarkChessPiece(PieceType.ELEPHANT, "red"))  # 左相
        red_pieces_list.append(DarkChessPiece(PieceType.ADVISOR, "red"))   # 左士
        # (9, 4) 是帅的位置，已经放置
        red_pieces_list.append(DarkChessPiece(PieceType.ADVISOR, "red"))   # 右士
        red_pieces_list.append(DarkChessPiece(PieceType.ELEPHANT, "red"))  # 右相
        red_pieces_list.append(DarkChessPiece(PieceType.HORSE, "red"))     # 右马
        red_pieces_list.append(DarkChessPiece(PieceType.CHARIOT, "red"))   # 右车
        
        # 第二排（row=7）：炮
        red_pieces_list.append(DarkChessPiece(PieceType.CANNON, "red"))    # 左炮
        red_pieces_list.append(DarkChessPiece(PieceType.CANNON, "red"))    # 右炮
        
        # 第三排（row=6）：兵
        for _ in range(5):
            red_pieces_list.append(DarkChessPiece(PieceType.SOLDIER, "red"))
        
        # 设置所有红方棋子为隐藏状态
        for piece in red_pieces_list:
            piece.state = PieceState.HIDDEN
        
        # 随机打乱红方棋子
        random.shuffle(red_pieces_list)
        
        # 红方棋子的位置（除了帅的位置）
        red_positions = [
            (9, 0), (9, 1), (9, 2), (9, 3),  # 第一排左半部分
            # (9, 4) 是帅的位置
            (9, 5), (9, 6), (9, 7), (9, 8),  # 第一排右半部分
            (7, 1), (7, 7),                   # 第二排两个炮
            (6, 0), (6, 2), (6, 4), (6, 6), (6, 8)  # 第三排五个兵
        ]
        
        # 将随机打乱的红方棋子放置到这些位置
        for i, piece in enumerate(red_pieces_list):
            row, col = red_positions[i]
            self.set_piece(row, col, piece)
        
        # 黑方棋子按照传统象棋位置摆放，但除了将之外随机交换位置（背面朝上）
        # 创建黑方所有棋子（除将帅）
        black_pieces_list = []
        
        # 第一排（row=0）：车马象士将士象马车（除了将）
        black_pieces_list.append(DarkChessPiece(PieceType.CHARIOT, "black"))   # 左车
        black_pieces_list.append(DarkChessPiece(PieceType.HORSE, "black"))     # 左马
        black_pieces_list.append(DarkChessPiece(PieceType.ELEPHANT, "black"))  # 左象
        black_pieces_list.append(DarkChessPiece(PieceType.ADVISOR, "black"))   # 左士
        # (0, 4) 是将的位置，已经放置
        black_pieces_list.append(DarkChessPiece(PieceType.ADVISOR, "black"))   # 右士
        black_pieces_list.append(DarkChessPiece(PieceType.ELEPHANT, "black"))  # 右象
        black_pieces_list.append(DarkChessPiece(PieceType.HORSE, "black"))     # 右马
        black_pieces_list.append(DarkChessPiece(PieceType.CHARIOT, "black"))   # 右车
        
        # 第二排（row=2）：炮
        black_pieces_list.append(DarkChessPiece(PieceType.CANNON, "black"))    # 左炮
        black_pieces_list.append(DarkChessPiece(PieceType.CANNON, "black"))    # 右炮
        
        # 第三排（row=3）：卒
        for _ in range(5):
            black_pieces_list.append(DarkChessPiece(PieceType.SOLDIER, "black"))
        
        # 设置所有黑方棋子为隐藏状态
        for piece in black_pieces_list:
            piece.state = PieceState.HIDDEN
        
        # 随机打乱黑方棋子
        random.shuffle(black_pieces_list)
        
        # 黑方棋子的位置（除了将的位置）
        black_positions = [
            (0, 0), (0, 1), (0, 2), (0, 3),  # 第一排左半部分
            # (0, 4) 是将的位置
            (0, 5), (0, 6), (0, 7), (0, 8),  # 第一排右半部分
            (2, 1), (2, 7),                   # 第二排两个炮
            (3, 0), (3, 2), (3, 4), (3, 6), (3, 8)  # 第三排五个卒
        ]
        
        # 将随机打乱的黑方棋子放置到这些位置
        for i, piece in enumerate(black_pieces_list):
            row, col = black_positions[i]
            self.set_piece(row, col, piece)
                
    def get_piece(self, row: int, col: int) -> Optional[DarkChessPiece]:
        """获取指定位置的棋子"""
        if self.is_valid_position(row, col):
            return self.board[row][col]
        return None
        
    def set_piece(self, row: int, col: int, piece: Optional[DarkChessPiece]):
        """设置指定位置的棋子"""
        if self.is_valid_position(row, col):
            self.board[row][col] = piece
            if piece:
                piece.position = (row, col)
                
    def is_valid_position(self, row: int, col: int) -> bool:
        """检查位置是否有效"""
        return 0 <= row < 10 and 0 <= col < 9
        
    def move_piece(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> Optional[DarkChessPiece]:
        """移动棋子，移动时自动翻开，返回被吃掉的棋子"""
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        
        piece = self.get_piece(from_row, from_col)
        captured_piece = self.get_piece(to_row, to_col)
        
        if piece:
            # 移动时自动翻开
            if not piece.is_revealed():
                piece.reveal()
                
            self.set_piece(from_row, from_col, None)
            self.set_piece(to_row, to_col, piece)
            
        return captured_piece
        
    def is_valid_move(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int], player_color: str) -> bool:
        """验证移动是否合法"""
        from_row, from_col = from_pos
        piece = self.get_piece(from_row, from_col)
        
        if not piece:
            return False
            
        if piece.color != player_color:
            return False
            
        valid_moves = piece.get_valid_moves(self, from_pos)
        return to_pos in valid_moves
        
    def get_all_pieces(self, color: Optional[str] = None) -> List[Tuple[DarkChessPiece, Tuple[int, int]]]:
        """获取所有棋子"""
        pieces = []
        for row in range(10):
            for col in range(9):
                piece = self.get_piece(row, col)
                if piece:
                    if color is None or piece.color == color:
                        pieces.append((piece, (row, col)))
        return pieces
        
    def is_general_captured(self, color: str) -> bool:
        """检查将是否被吃掉"""
        for row in range(10):
            for col in range(9):
                piece = self.get_piece(row, col)
                if (piece and piece.color == color and 
                    piece.piece_type == PieceType.GENERAL):
                    return False
        return True