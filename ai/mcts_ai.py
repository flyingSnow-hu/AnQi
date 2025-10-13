"""
基于蒙特卡洛树搜索的AI实现
使用确定化（Determinization）方法处理不完全信息
"""
import random
import math
import copy
from typing import Tuple, List, Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.dark_chess_piece import DarkChessPiece, PieceType, PieceState
from game.dark_chess_board import DarkChessBoard
from ai.ai_player import AIPlayer


class MCTSNode:
    """MCTS树节点"""
    
    def __init__(self, parent=None, move=None):
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0.0
        self.visits = 0
        self.untried_moves = []
        
    def uct_value(self, exploration_weight=1.414):
        """UCT值计算"""
        if self.visits == 0:
            return float('inf')
        exploitation = self.wins / self.visits
        exploration = exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration
    
    def best_child(self, exploration_weight=1.414):
        """选择最佳子节点"""
        return max(self.children, key=lambda c: c.uct_value(exploration_weight))
    
    def most_visited_child(self):
        """选择访问次数最多的子节点"""
        return max(self.children, key=lambda c: c.visits)


class MCTSAI(AIPlayer):
    """基于MCTS的强化AI - 改进版"""
    
    def __init__(self, color: str, simulations=2000, max_depth=60):
        """
        初始化MCTS AI
        color: AI颜色
        simulations: 每次决策的模拟次数（增加到2000）
        max_depth: 最大搜索深度
        """
        super().__init__(color)
        self.simulations = simulations
        self.max_depth = max_depth
        
        # 明子基础价值表
        self.base_piece_values = {
            PieceType.GENERAL: 10000,
            PieceType.CHARIOT: 120,
            PieceType.CANNON: 55,
            PieceType.HORSE: 50,
            PieceType.ADVISOR: 30,
            PieceType.ELEPHANT: 28,
            PieceType.SOLDIER: 20
        }
        
        # 暗子期望价值（未翻开的棋子，基于概率期望）
        self.hidden_piece_values = {
            PieceType.CHARIOT: 80,
            PieceType.CANNON: 40,
            PieceType.HORSE: 38,
            PieceType.ADVISOR: 22,
            PieceType.ELEPHANT: 20,
            PieceType.SOLDIER: 15
        }
        
        # 暗子平均期望价值
        self.average_hidden_value = 32
        
    def _get_piece_value(self, piece, row=None, col=None):
        """获取棋子价值（明子考虑位置加成，暗子使用期望值）"""
        if piece.is_revealed():
            # 明子：基础价值 + 位置加成
            base_value = self.base_piece_values.get(piece.piece_type, 0)
            
            if row is not None and col is not None:
                position_bonus = self._get_position_bonus(piece, row, col)
                return base_value + position_bonus
            else:
                return base_value
        else:
            # 暗子：期望价值（不考虑位置）
            if piece.position:
                inferred_type = piece._get_piece_type_by_position(*piece.position)
                return self.hidden_piece_values.get(inferred_type, self.average_hidden_value)
            else:
                return self.average_hidden_value
    
    def _get_position_bonus(self, piece, row, col):
        """计算明子的位置加成"""
        bonus = 0
        piece_type = piece.piece_type
        color = piece.color
        
        # 中心距离（距离棋盘中心的曼哈顿距离）
        center_dist = abs(row - 4.5) + abs(col - 4)
        
        if piece_type == PieceType.GENERAL:
            # 将帅位置加成
            if col == 4:  # 在中心列最安全
                bonus += 20
            else:
                bonus -= 5
            
            # 保持在后排
            if (color == "red" and row >= 8) or (color == "black" and row <= 1):
                bonus += 15
            else:
                bonus -= 10
        
        elif piece_type == PieceType.ADVISOR:
            # 士位置加成 - 靠近将帅
            if (color == "red" and row >= 7 and 3 <= col <= 5) or \
               (color == "black" and row <= 2 and 3 <= col <= 5):
                bonus += 10
            
            # 在将帅旁边
            general_row = 9 if color == "red" else 0
            if row == general_row and abs(col - 4) == 1:
                bonus += 15
        
        elif piece_type == PieceType.ELEPHANT:
            # 相位置加成 - 防守本方阵地
            if (color == "red" and row >= 5) or (color == "black" and row <= 4):
                bonus += 8
            else:
                bonus -= 5  # 过河降价值
            
            # 控制中心线
            if col == 4:
                bonus += 5
        
        elif piece_type == PieceType.CHARIOT:
            # 车位置加成 - 控制整条线
            # 在中心区域价值高
            bonus += int((10 - center_dist) * 3)
            
            # 前进到对方阵地
            if (color == "red" and row <= 4) or (color == "black" and row >= 5):
                bonus += 25
            
            # 占据中路
            if 3 <= col <= 5:
                bonus += 15
        
        elif piece_type == PieceType.CANNON:
            # 炮位置加成 - 控制中心
            bonus += int((10 - center_dist) * 2)
            
            # 前进到对方阵地
            if (color == "red" and row <= 4) or (color == "black" and row >= 5):
                bonus += 20
            
            # 占据要道
            if col in [2, 4, 6]:
                bonus += 10
        
        elif piece_type == PieceType.HORSE:
            # 马位置加成 - 控制中心
            bonus += int((10 - center_dist) * 2.5)
            
            # 前进到中心区域
            if 3 <= row <= 6:
                bonus += 20
            
            # 在对方阵地活跃
            if (color == "red" and row <= 3) or (color == "black" and row >= 6):
                bonus += 15
        
        elif piece_type == PieceType.SOLDIER:
            # 兵位置加成 - 前进奖励
            if color == "red":
                # 红兵向上走
                advance = 9 - row
                bonus += advance * 5
                
                # 过河大幅加分
                if row < 5:
                    bonus += 30
                
                # 深入敌阵
                if row <= 2:
                    bonus += 40
                
                # 逼近对方将帅
                if row <= 1:
                    bonus += 50
            else:
                # 黑卒向下走
                advance = row
                bonus += advance * 5
                
                # 过河大幅加分
                if row > 4:
                    bonus += 30
                
                # 深入敌阵
                if row >= 7:
                    bonus += 40
                
                # 逼近对方将帅
                if row >= 8:
                    bonus += 50
            
            # 中路兵价值更高
            if 3 <= col <= 5:
                bonus += 10
        
        return bonus

    def get_move(self, board, game_state) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """获取AI的移动决策"""
        legal_moves = self.get_all_legal_moves(board, self.color)
        
        if not legal_moves:
            return None
        
        if len(legal_moves) == 1:
            return legal_moves[0]
        
        # 立即吃将的移动
        for from_pos, to_pos in legal_moves:
            target = board.get_piece(*to_pos)
            if target and target.piece_type == PieceType.GENERAL:
                return (from_pos, to_pos)
        
        # 检查己方将帅是否被攻击，优先解围
        escape_move = self._find_escape_move(board, legal_moves)
        if escape_move:
            return escape_move
        
        # 运行增强的MCTS搜索
        best_move = self._enhanced_mcts_search(board, game_state, legal_moves)
        
        if not best_move:
            best_move = self._heuristic_move(board, game_state, legal_moves)
        
        return best_move
    
    def _find_escape_move(self, board, legal_moves):
        """查找解围移动 - 当将帅被攻击时优先解围"""
        # 找到己方将帅位置
        general_pos = None
        for row in range(10):
            for col in range(9):
                piece = board.get_piece(row, col)
                if piece and piece.color == self.color and piece.piece_type == PieceType.GENERAL:
                    general_pos = (row, col)
                    break
            if general_pos:
                break
        
        if not general_pos:
            return None
        
        # 检查将帅是否被攻击
        if not self._is_general_under_attack(board, general_pos):
            return None
        
        # 将帅被攻击！评估所有解围方案
        escape_moves = []
        
        for from_pos, to_pos in legal_moves:
            # 模拟这一步
            sim_board = self._copy_board(board)
            piece = sim_board.get_piece(*from_pos)
            target = sim_board.get_piece(*to_pos)
            
            sim_board.set_piece(*to_pos, piece)
            sim_board.set_piece(*from_pos, None)
            
            # 检查移动后将帅是否还在被攻击
            new_general_pos = to_pos if from_pos == general_pos else general_pos
            
            if not self._is_general_under_attack(sim_board, new_general_pos):
                # 这个移动可以解围
                score = 10000  # 基础解围分数
                
                # 如果是吃掉攻击者，加分
                if target:
                    score += 5000
                
                # 如果是将帅自己移动，略微降分（优先用其他子保护）
                if from_pos == general_pos:
                    score -= 1000
                
                escape_moves.append((from_pos, to_pos, score))
        
        if escape_moves:
            # 选择最佳解围方案
            escape_moves.sort(key=lambda x: x[2], reverse=True)
            return (escape_moves[0][0], escape_moves[0][1])
        
        return None
    
    def _is_general_under_attack(self, board, general_pos):
        """检查将帅是否被攻击"""
        if not general_pos:
            return False
        
        # 获取对方所有棋子的攻击范围
        opponent_color = self.opponent_color
        
        for row in range(10):
            for col in range(9):
                piece = board.get_piece(row, col)
                if piece and piece.color == opponent_color:
                    valid_moves = piece.get_valid_moves(board, (row, col))
                    if general_pos in valid_moves:
                        return True
        
        return False
    
    def _enhanced_mcts_search(self, board, game_state, legal_moves):
        """增强的MCTS搜索"""
        root = MCTSNode()
        
        # 预先评估所有移动，优先尝试好的移动
        move_scores = []
        for move in legal_moves:
            score = self._quick_evaluate_move(board, move)
            move_scores.append((move, score))
        
        # 按评分排序，但保留一定随机性
        move_scores.sort(key=lambda x: x[1], reverse=True)
        sorted_moves = [m for m, s in move_scores]
        
        # 前70%的好移动 + 30%随机性
        good_moves_count = max(1, int(len(sorted_moves) * 0.7))
        root.untried_moves = sorted_moves[:good_moves_count] + random.sample(
            sorted_moves[good_moves_count:], 
            min(len(sorted_moves) - good_moves_count, max(1, len(sorted_moves) // 3))
        )
        random.shuffle(root.untried_moves)
        
        for iteration in range(self.simulations):
            # 1. Selection
            node = root
            sim_board = self._copy_board(board)
            sim_state = self._copy_game_state(game_state)
            depth = 0
            
            while not node.untried_moves and node.children and depth < self.max_depth:
                node = node.best_child()
                if node.move:
                    self._apply_move(sim_board, sim_state, node.move)
                    depth += 1
            
            # 2. Expansion
            if node.untried_moves and depth < self.max_depth:
                move = node.untried_moves.pop(0)  # 取第一个（已排序）
                self._apply_move(sim_board, sim_state, move)
                child = MCTSNode(parent=node, move=move)
                
                child_moves = self.get_all_legal_moves(sim_board, sim_state.current_turn)
                # 子节点也进行排序
                child_move_scores = [(m, self._quick_evaluate_move(sim_board, m)) for m in child_moves]
                child_move_scores.sort(key=lambda x: x[1], reverse=True)
                child.untried_moves = [m for m, s in child_move_scores[:min(20, len(child_move_scores))]]
                
                node.children.append(child)
                node = child
                depth += 1
            
            # 3. Simulation（使用更智能的策略）
            reward = self._enhanced_simulate(sim_board, sim_state, self.max_depth - depth)
            
            # 4. Backpropagation
            while node is not None:
                node.visits += 1
                node.wins += reward if node.parent and node.parent.move else -reward
                node = node.parent
        
        # 选择最佳移动（结合访问次数和胜率）
        if root.children:
            best_child = max(root.children, key=lambda c: c.wins / c.visits if c.visits > 0 else -float('inf'))
            return best_child.move
        
        return None
    
    def _quick_evaluate_move(self, board, move):
        """快速评估一个移动的价值"""
        from_pos, to_pos = move
        piece = board.get_piece(*from_pos)
        target = board.get_piece(*to_pos)
        
        score = 0.0
        
        # 1. 吃子价值（大幅提升权重）
        if target:
            if target.piece_type == PieceType.GENERAL:
                return 100000.0
            
            # 获取目标棋子的完整价值（包括位置加成）
            target_value = self._get_piece_value(target, to_pos[0], to_pos[1])
            # 己方棋子的价值（用于评估交换）
            piece_value = self._get_piece_value(piece, from_pos[0], from_pos[1])
            
            # 吃子基础奖励（大幅提升）
            score += target_value * 10  # 从1.5/3倍提升到10倍
            
            # 用低价值子吃高价值子额外奖励
            if target_value > piece_value * 0.8:  # 吃价值相近或更高的子
                score += target_value * 5  # 额外奖励
        
        # 2. 位置价值改进（降低权重，让吃子更优先）
        if piece and piece.is_revealed():
            # 如果是明子，计算位置改进
            from_value = self._get_piece_value(piece, from_pos[0], from_pos[1])
            to_value = self._get_piece_value(piece, to_pos[0], to_pos[1])
            position_gain = to_value - from_value
            score += position_gain * 0.5  # 位置改进权重降低
        else:
            # 暗子简化评估
            from_center = abs(from_pos[0] - 4.5) + abs(from_pos[1] - 4)
            to_center = abs(to_pos[0] - 4.5) + abs(to_pos[1] - 4)
            score += (from_center - to_center) * 2
        
        # 3. 兵卒前进（降低权重）
        if piece:
            if piece.piece_type == PieceType.SOLDIER or (
                not piece.is_revealed() and piece._get_piece_type_by_position(*from_pos) == PieceType.SOLDIER
            ):
                if (self.color == "red" and to_pos[0] < from_pos[0]) or \
                   (self.color == "black" and to_pos[0] > from_pos[0]):
                    score += 8  # 从15降低到8
        
            # 车炮马优先发展（降低权重）
            if not piece.is_revealed():
                inferred = piece._get_piece_type_by_position(*from_pos)
                if inferred in [PieceType.CHARIOT, PieceType.CANNON, PieceType.HORSE]:
                    score += 5  # 从12降低到5
        
        return score
    
    def _enhanced_simulate(self, board, game_state, max_moves=40):
        """增强的模拟策略"""
        moves_count = 0
        
        while not game_state.game_over and moves_count < max_moves:
            legal_moves = self.get_all_legal_moves(board, game_state.current_turn)
            if not legal_moves:
                break
            
            move = self._very_smart_rollout(board, game_state, legal_moves)
            if move:
                self._apply_move(board, game_state, move)
            moves_count += 1
        
        return self._enhanced_evaluate(board, game_state)
    
    def _very_smart_rollout(self, board, game_state, legal_moves):
        """非常智能的rollout策略"""
        if not legal_moves:
            return None
        
        # 1. 立即获胜
        for from_pos, to_pos in legal_moves:
            target = board.get_piece(*to_pos)
            if target and target.piece_type == PieceType.GENERAL:
                return (from_pos, to_pos)
        
        # 2. 评估所有移动
        move_evals = []
        for move in legal_moves:
            score = self._quick_evaluate_move(board, move)
            move_evals.append((move, score))
        
        # 3. 按评分分组
        move_evals.sort(key=lambda x: x[1], reverse=True)
        
        # 4. 80%选择前3名，20%随机
        if random.random() < 0.8 and len(move_evals) >= 3:
            return random.choice(move_evals[:3])[0]
        elif random.random() < 0.9 and len(move_evals) >= 5:
            return random.choice(move_evals[:5])[0]
        else:
            return move_evals[0][0]
    
    def _enhanced_evaluate(self, board, game_state):
        """增强的评估函数"""
        if game_state.game_over:
            winner_color = "红方" if self.color == "red" else "黑方"
            if game_state.winner == winner_color:
                return 10000.0
            else:
                return -10000.0
        
        score = 0.0
        my_pieces = []
        opp_pieces = []
        
        # 1. 物质 + 位置评估
        for row in range(10):
            for col in range(9):
                piece = board.get_piece(row, col)
                if not piece:
                    continue
                
                base_value = self._get_piece_value(piece, row, col)
                
                total_value = base_value
                
                if piece.color == self.color:
                    score += total_value
                    my_pieces.append(piece)
                else:
                    score -= total_value
                    opp_pieces.append(piece)
        
        # 2. 机动性评估（更重要）
        my_mobility = len(self.get_all_legal_moves(board, self.color))
        opp_mobility = len(self.get_all_legal_moves(board, self.opponent_color))
        score += (my_mobility - opp_mobility) * 5
        
        # 3. 棋子数量优势
        score += (len(my_pieces) - len(opp_pieces)) * 10
        
        # 4. 将帅安全性
        my_general_safe = self._is_general_safe(board, self.color)
        opp_general_safe = self._is_general_safe(board, self.opponent_color)
        if my_general_safe:
            score += 50
        if not opp_general_safe:
            score += 100
        
        return score
    
    def _is_general_safe(self, board, color):
        """检查将帅是否安全"""
        # 找到将帅
        general_pos = None
        for row in range(10):
            for col in range(9):
                piece = board.get_piece(row, col)
                if piece and piece.color == color and piece.piece_type == PieceType.GENERAL:
                    general_pos = (row, col)
                    break
            if general_pos:
                break
        
        if not general_pos:
            return False
        
        # 检查周围是否有保护
        protection_count = 0
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = general_pos[0] + dr, general_pos[1] + dc
            if 0 <= nr < 10 and 0 <= nc < 9:
                piece = board.get_piece(nr, nc)
                if piece and piece.color == color:
                    protection_count += 1
        
        return protection_count >= 1
    
    def _heuristic_move(self, board, game_state, legal_moves):
        """改进的启发式移动"""
        if not legal_moves:
            return None
        
        move_scores = []
        for move in legal_moves:
            score = self._quick_evaluate_move(board, move)
            move_scores.append((move, score))
        
        move_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 前3名中随机选择
        top_moves = move_scores[:min(3, len(move_scores))]
        return random.choice(top_moves)[0]
    
    def _copy_board(self, board):
        """复制棋盘"""
        new_board = DarkChessBoard()
        new_board.board = [[None for _ in range(9)] for _ in range(10)]
        
        for row in range(10):
            for col in range(9):
                piece = board.get_piece(row, col)
                if piece:
                    new_piece = DarkChessPiece(piece.piece_type, piece.color)
                    new_piece.state = piece.state
                    new_board.set_piece(row, col, new_piece)
        
        return new_board
    
    def _copy_game_state(self, game_state):
        """复制游戏状态"""
        new_state = type('GameState', (), {})()
        new_state.current_turn = game_state.current_turn
        new_state.move_count = game_state.move_count
        new_state.game_over = game_state.game_over
        new_state.winner = game_state.winner
        return new_state
    
    def _apply_move(self, board, game_state, move):
        """应用移动"""
        from_pos, to_pos = move
        piece = board.get_piece(*from_pos)
        if not piece:
            return
        
        target = board.get_piece(*to_pos)
        
        if target and target.piece_type == PieceType.GENERAL:
            game_state.game_over = True
            game_state.winner = "红方" if piece.color == "red" else "黑方"
        
        board.set_piece(*to_pos, piece)
        board.set_piece(*from_pos, None)
        
        game_state.current_turn = "black" if game_state.current_turn == "red" else "red"
        game_state.move_count += 1


class SimpleAI(AIPlayer):
    """简化版AI - 改进的启发式"""
    
    def __init__(self, color: str):
        super().__init__(color)
        self.piece_values = {
            PieceType.GENERAL: 10000,
            PieceType.CHARIOT: 100,
            PieceType.CANNON: 50,
            PieceType.HORSE: 45,
            PieceType.ADVISOR: 25,
            PieceType.ELEPHANT: 25,
            PieceType.SOLDIER: 18
        }
        
    def get_move(self, board, game_state):
        """获取移动"""
        legal_moves = self.get_all_legal_moves(board, self.color)
        
        if not legal_moves:
            return None
        
        if len(legal_moves) == 1:
            return legal_moves[0]
        
        # 立即吃将
        for from_pos, to_pos in legal_moves:
            target = board.get_piece(*to_pos)
            if target and target.piece_type == PieceType.GENERAL:
                return (from_pos, to_pos)
        
        return self._improved_heuristic(board, legal_moves)
    
    def _improved_heuristic(self, board, legal_moves):
        """改进的启发式"""
        best_move = None
        best_score = -float('inf')
        
        for from_pos, to_pos in legal_moves:
            score = 0
            piece = board.get_piece(*from_pos)
            target = board.get_piece(*to_pos)
            
            # 吃子
            if target:
                value = self.piece_values.get(target.piece_type, 0) if target.is_revealed() else 35
                score += value * 10
            
            # 中心控制
            center_dist = abs(to_pos[0] - 4.5) + abs(to_pos[1] - 4)
            score += (10 - center_dist) * 5
            
            # 前进
            if piece and piece.piece_type == PieceType.SOLDIER:
                if (self.color == "red" and to_pos[0] < from_pos[0]) or \
                   (self.color == "black" and to_pos[0] > from_pos[0]):
                    score += 20
            
            # 发展
            if piece and not piece.is_revealed():
                inferred = piece._get_piece_type_by_position(*from_pos)
                if inferred in [PieceType.CHARIOT, PieceType.CANNON, PieceType.HORSE]:
                    score += 15
            
            score += random.uniform(0, 10)
            
            if score > best_score:
                best_score = score
                best_move = (from_pos, to_pos)
        
        return best_move
