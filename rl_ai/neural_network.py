"""
揭棋游戏的神经网络模型
使用卷积神经网络提取棋盘特征
输出策略（移动概率）和价值（局面评估）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DarkChessNet(nn.Module):
    """
    揭棋神经网络
    输入：棋盘状态（多通道特征图）
    输出：
        - 策略：每个合法移动的概率分布
        - 价值：当前局面的胜率评估 [-1, 1]
    """
    
    def __init__(self, num_channels=128, num_res_blocks=10):
        """
        初始化网络
        num_channels: 卷积层通道数
        num_res_blocks: 残差块数量
        """
        super(DarkChessNet, self).__init__()
        
        # 输入特征通道数：
        # 红方明子(7) + 红方暗子(1) + 黑方明子(7) + 黑方暗子(1) + 
        # 当前玩家(1) + 回合数(1) = 18通道
        self.input_channels = 18
        
        # 初始卷积层
        self.conv_input = nn.Conv2d(self.input_channels, num_channels, 
                                     kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_channels)
        
        # 残差块
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])
        
        # 策略头（Policy Head）
        self.policy_conv = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        # 10x9棋盘，每个位置可以移动到其他位置，简化为 10*9*10*9 = 8100
        self.policy_fc = nn.Linear(32 * 10 * 9, 10 * 9 * 10 * 9)
        
        # 价值头（Value Head）
        self.value_conv = nn.Conv2d(num_channels, 8, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(8)
        self.value_fc1 = nn.Linear(8 * 10 * 9, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
    def forward(self, x):
        """
        前向传播
        x: (batch_size, 18, 10, 9) - 棋盘状态张量
        返回: (policy_logits, value)
        """
        # 初始卷积
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # 残差块
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # 策略头
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)  # Flatten
        policy = self.policy_fc(policy)
        
        # 价值头
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)  # Flatten
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value
    
    def predict(self, board_state):
        """
        预测单个棋盘状态
        board_state: numpy array (18, 10, 9)
        返回: (policy_probs, value)
        """
        self.eval()
        with torch.no_grad():
            # 转换为tensor并添加batch维度，移动到模型所在设备
            device = next(self.parameters()).device
            x = torch.FloatTensor(board_state).unsqueeze(0).to(device)
            policy_logits, value = self.forward(x)
            policy_probs = F.softmax(policy_logits, dim=1)
            return policy_probs.squeeze(0).cpu().numpy(), value.item()


class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, 
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


class BoardEncoder:
    """棋盘状态编码器 - 将棋盘转换为神经网络输入"""
    
    @staticmethod
    def encode_board(board, current_color):
        """
        将棋盘编码为18通道的特征张量
        
        通道说明：
        0-6: 红方明子 (将、车、马、炮、士、相、兵)
        7: 红方暗子
        8-14: 黑方明子
        15: 黑方暗子
        16: 当前玩家 (1=红方, 0=黑方)
        17: 回合进度 (归一化到0-1)
        
        返回: numpy array (18, 10, 9)
        """
        from game.dark_chess_piece import PieceType
        
        state = np.zeros((18, 10, 9), dtype=np.float32)
        
        # 棋子类型到通道的映射
        piece_type_to_channel = {
            PieceType.GENERAL: 0,
            PieceType.CHARIOT: 1,
            PieceType.HORSE: 2,
            PieceType.CANNON: 3,
            PieceType.ADVISOR: 4,
            PieceType.ELEPHANT: 5,
            PieceType.SOLDIER: 6
        }
        
        # 编码棋盘上的棋子
        for row in range(10):
            for col in range(9):
                piece = board.get_piece(row, col)
                if piece:
                    if piece.is_revealed():
                        # 明子
                        channel_offset = 0 if piece.color == "red" else 8
                        channel = channel_offset + piece_type_to_channel.get(piece.piece_type, 6)
                        state[channel, row, col] = 1.0
                    else:
                        # 暗子
                        channel = 7 if piece.color == "red" else 15
                        state[channel, row, col] = 1.0
        
        # 当前玩家
        state[16, :, :] = 1.0 if current_color == "red" else 0.0
        
        # 回合进度（假设最多200回合）
        # 这个需要从game_state获取，这里先简化为0
        state[17, :, :] = 0.0
        
        return state
    
    @staticmethod
    def encode_move(from_pos, to_pos):
        """
        将移动编码为策略输出的索引
        from_pos: (row, col)
        to_pos: (row, col)
        返回: 0-8099之间的整数
        """
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        return from_row * 9 * 10 * 9 + from_col * 10 * 9 + to_row * 9 + to_col
    
    @staticmethod
    def decode_move(move_idx):
        """
        将策略输出索引解码为移动
        move_idx: 0-8099之间的整数
        返回: ((from_row, from_col), (to_row, to_col))
        """
        from_row = move_idx // (9 * 10 * 9)
        remainder = move_idx % (9 * 10 * 9)
        from_col = remainder // (10 * 9)
        remainder = remainder % (10 * 9)
        to_row = remainder // 9
        to_col = remainder % 9
        return ((from_row, from_col), (to_row, to_col))
