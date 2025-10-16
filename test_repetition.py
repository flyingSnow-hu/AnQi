#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试三次重复局面判和功能
"""
import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from game.game_engine import GameEngine
from players.human_player import HumanPlayer


def test_repetition():
    """测试三次重复判和"""
    print("="*60)
    print("测试三次重复局面判和功能")
    print("="*60)
    
    # 创建游戏引擎
    red_player = HumanPlayer("红方测试", "red")
    black_player = HumanPlayer("黑方测试", "black")
    engine = GameEngine(red_player, black_player)
    
    print("\n初始局面哈希:", len(engine.game_state.position_history), "个")
    print("初始重复次数:", engine.game_state.repeat_count)
    
    # 模拟一个会导致重复的走法序列
    # 假设红方和黑方各走一步,然后撤回,重复3次
    moves = [
        # 第一轮
        ((0, 0), (1, 0)),  # 红方移动
        ((9, 0), (8, 0)),  # 黑方移动
        ((1, 0), (0, 0)),  # 红方撤回
        ((8, 0), (9, 0)),  # 黑方撤回
        # 第二轮
        ((0, 0), (1, 0)),  # 红方移动
        ((9, 0), (8, 0)),  # 黑方移动
        ((1, 0), (0, 0)),  # 红方撤回
        ((8, 0), (9, 0)),  # 黑方撤回
        # 第三轮
        ((0, 0), (1, 0)),  # 红方移动
        ((9, 0), (8, 0)),  # 黑方移动
        ((1, 0), (0, 0)),  # 红方撤回
        ((8, 0), (9, 0)),  # 黑方撤回(应该触发三次重复)
    ]
    
    print("\n开始测试移动序列...")
    for i, (from_pos, to_pos) in enumerate(moves):
        if engine.game_state.game_over:
            print(f"\n游戏在第{i}步结束")
            if engine.game_state.is_draw_by_repetition:
                print("✓ 三次重复判和触发成功!")
            break
        
        # 尝试移动
        piece = engine.board.get_piece(*from_pos)
        if piece:
            print(f"\n第{i+1}步: {from_pos} -> {to_pos}")
            success = engine.make_move(from_pos, to_pos)
            if success:
                print(f"  当前局面哈希数量: {len(engine.game_state.position_history)}")
                print(f"  重复次数: {engine.game_state.repeat_count}")
                print(f"  重复惩罚: {engine.game_state.repetition_penalty}")
            else:
                print(f"  移动失败(可能是非法移动)")
        else:
            print(f"\n第{i+1}步: {from_pos} 位置没有棋子,跳过")
    
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)
    print(f"游戏是否结束: {engine.game_state.game_over}")
    print(f"是否三次重复: {engine.game_state.is_draw_by_repetition}")
    print(f"总局面数: {len(engine.game_state.position_history)}")
    print(f"不同局面数: {len(engine.game_state.repeat_count)}")
    

if __name__ == "__main__":
    test_repetition()
