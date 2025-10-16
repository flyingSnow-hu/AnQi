#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试长捉检测功能
"""
import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from game.game_engine import GameEngine
from players.human_player import HumanPlayer


def test_perpetual_chase():
    """测试长捉检测"""
    print("="*60)
    print("测试长捉检测功能")
    print("="*60)
    
    # 创建游戏引擎
    red_player = HumanPlayer("红方测试", "red")
    black_player = HumanPlayer("黑方测试", "black")
    engine = GameEngine(red_player, black_player)
    
    print("\n初始状态:")
    print(f"  连续追捉次数: {engine.game_state.consecutive_chases}")
    print(f"  追捉方: {engine.game_state.chasing_side}")
    print(f"  被追捉棋子位置: {engine.game_state.chased_piece_pos}")
    
    # 测试威胁检测
    print("\n测试1: 棋子威胁检测")
    
    # 遍历棋盘找一些棋子测试
    test_count = 0
    for row in range(10):
        for col in range(9):
            piece = engine.board.get_piece(row, col)
            if piece and piece.is_revealed() and test_count < 3:
                pos = (row, col)
                opponent_color = "black" if piece.color == "red" else "red"
                
                is_threatened = engine.is_piece_under_threat(pos, opponent_color)
                print(f"  位置{pos}的{piece.color}{piece.piece_type.value}: "
                      f"被{opponent_color}威胁={is_threatened}")
                
                if is_threatened:
                    can_escape = engine.can_piece_escape(pos, opponent_color)
                    print(f"    能否逃脱: {can_escape}")
                
                test_count += 1
    
    # 测试2: 模拟追捉场景
    print("\n测试2: 模拟追捉场景")
    print("  注意: 由于初始棋盘是随机的,需要实际对局才能测试长捉")
    print("  这里只测试计数机制")
    
    # 手动更新追捉状态,模拟连续追捉同一个棋子
    test_pos = (5, 5)
    for i in range(7):
        is_perpetual = engine.game_state.update_chase_status(True, "red", test_pos)
        print(f"  第{i+1}次追捉位置{test_pos}: "
              f"连续次数={engine.game_state.consecutive_chases}, "
              f"触发长捉={is_perpetual}")
        
        if is_perpetual:
            print(f"\n  ✓ 长捉判负触发!")
            print(f"    判负方: 红方")
            print(f"    获胜方: {engine.game_state.winner}")
            break
    
    # 测试3: 切换目标重置机制
    print("\n测试3: 切换追捉目标后重置计数")
    engine.game_state.update_chase_status(True, "red", (5, 5))
    print(f"  红方追捉(5,5)第1次: {engine.game_state.consecutive_chases}")
    
    engine.game_state.update_chase_status(True, "red", (5, 5))
    print(f"  红方追捉(5,5)第2次: {engine.game_state.consecutive_chases}")
    
    # 切换目标,应该重置计数
    engine.game_state.update_chase_status(True, "red", (6, 6))
    print(f"  红方追捉(6,6)(切换目标): {engine.game_state.consecutive_chases}")
    
    # 黑方追捉,应该重置
    engine.game_state.update_chase_status(True, "black", (5, 5))
    print(f"  黑方追捉(5,5)(切换追捉方): {engine.game_state.consecutive_chases}")
    
    # 没有追捉,应该重置
    engine.game_state.update_chase_status(False, None, None)
    print(f"  没有追捉(重置): {engine.game_state.consecutive_chases}")
    
    # 测试4: 长捉惩罚
    print("\n测试4: 长捉惩罚值")
    for i in range(7):
        engine.game_state.update_chase_status(True, "red", (5, 5))
        penalty = engine.game_state.get_perpetual_chase_penalty()
        print(f"  连续{engine.game_state.consecutive_chases}次追捉, 惩罚值: {penalty}")
    
    # 测试5: 查找被追捉棋子
    print("\n测试5: 查找被追捉的棋子")
    chased_red = engine.find_chased_piece("black")
    chased_black = engine.find_chased_piece("red")
    print(f"  黑方追捉的红方棋子: {chased_red}")
    print(f"  红方追捉的黑方棋子: {chased_black}")
    
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)
    

if __name__ == "__main__":
    test_perpetual_chase()
