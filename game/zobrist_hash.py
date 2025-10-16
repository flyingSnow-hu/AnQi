"""
Zobrist哈希 - 用于快速判断局面重复
"""
import random


class ZobristHash:
    """Zobrist哈希,用于快速判断局面重复"""
    
    def __init__(self, seed=12345):
        """
        初始化Zobrist哈希表
        seed: 随机数种子,保证可复现
        """
        random.seed(seed)
        
        # 为每个位置(10x9=90)的每种棋子状态生成随机数
        # 棋子状态: 红方7种+黑方7种+空=15种
        self.piece_keys = {}
        
        # 棋子类型(用piece_type的value值)
        piece_types = [
            '將', '仕', '相', '俥', '傌', '炮', '兵',  # 红方
            '帥', '士', '象', '車', '馬', '砲', '卒',  # 黑方
            'dark_red', 'dark_black'  # 暗棋
        ]
        
        # 为每个位置的每种状态生成64位随机数
        for row in range(10):
            for col in range(9):
                pos = (row, col)
                self.piece_keys[pos] = {}
                for piece_type in piece_types:
                    self.piece_keys[pos][piece_type] = random.getrandbits(64)
                # 空位也需要一个键
                self.piece_keys[pos][None] = random.getrandbits(64)
        
        # 行棋方标记(红方/黑方)
        self.side_to_move_key = random.getrandbits(64)
    
    def compute_hash(self, board, current_player):
        """
        计算当前局面的哈希值
        
        Args:
            board: DarkChessBoard对象
            current_player: 当前行棋方('red'或'black')
        
        Returns:
            64位整数哈希值
        """
        hash_value = 0
        
        # 遍历棋盘所有位置
        for row in range(10):
            for col in range(9):
                piece = board.get_piece(row, col)
                pos = (row, col)
                
                if piece is None:
                    # 空位
                    hash_value ^= self.piece_keys[pos][None]
                elif not piece.is_revealed():
                    # 暗棋
                    key = 'dark_red' if piece.color == 'red' else 'dark_black'
                    hash_value ^= self.piece_keys[pos][key]
                else:
                    # 明棋 - 使用piece_type的value
                    piece_key = piece.piece_type.value
                    if pos in self.piece_keys and piece_key in self.piece_keys[pos]:
                        hash_value ^= self.piece_keys[pos][piece_key]
        
        # 如果是黑方行棋,异或行棋方标记
        if current_player == 'black':
            hash_value ^= self.side_to_move_key
        
        return hash_value
