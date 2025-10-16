#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试长将检测功能
"""
import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from game.game_engine import GameEngine
from players.human_player import HumanPlayer


def test_perpetual_check():
    """测试长将检测"""
    print("="*60)
    print("测试长将检测功能")
    print("="*60)
    
    # 创建游戏引擎
    red_player = HumanPlayer("红方测试", "red")
    black_player = HumanPlayer("黑方测试", "black")
    engine = GameEngine(red_player, black_player)
    
    print("\n初始状态:")
    print(f"  连续将军次数: {engine.game_state.consecutive_checks}")
    print(f"  将军方: {engine.game_state.checking_side}")
    
    # 测试将军检测
    print("\n测试1: 检查将军检测功能")
    
    # 找到红方的将帅位置
    red_general_pos = engine._find_general("red")
    black_general_pos = engine._find_general("black")
    
    print(f"  红方将帅位置: {red_general_pos}")
    print(f"  黑方将帅位置: {black_general_pos}")
    
    # 检查初始状态是否有将军
    red_in_check = engine.is_in_check("red")
    black_in_check = engine.is_in_check("black")
    
    print(f"  红方是否被将军: {red_in_check}")
    print(f"  黑方是否被将军: {black_in_check}")
    
    # 测试2: 模拟连续将军
    print("\n测试2: 模拟连续将军场景")
    print("  注意: 由于初始棋盘是随机的,需要实际对局才能测试长将")
    print("  这里只测试计数机制")
    
    # 手动更新将军状态,模拟连续将军
    for i in range(7):
        is_perpetual = engine.game_state.update_check_status(True, "red")
        print(f"  第{i+1}次将军: 连续次数={engine.game_state.consecutive_checks}, 触发长将={is_perpetual}")
        
        if is_perpetual:
            print(f"\n  ✓ 长将判负触发!")
            print(f"    判负方: 红方")
            print(f"    获胜方: {engine.game_state.winner}")
            break
    
    # 测试3: 重置机制
    print("\n测试3: 将军中断后重置计数")
    engine.game_state.update_check_status(True, "red")
    print(f"  红方第1次将军: {engine.game_state.consecutive_checks}")
    
    engine.game_state.update_check_status(True, "red")
    print(f"  红方第2次将军: {engine.game_state.consecutive_checks}")
    
    # 黑方将军,应该重置计数
    engine.game_state.update_check_status(True, "black")
    print(f"  黑方将军(切换): {engine.game_state.consecutive_checks}")
    
    # 没有将军,应该重置
    engine.game_state.update_check_status(False, None)
    print(f"  没有将军(重置): {engine.game_state.consecutive_checks}")
    
    # 测试4: 长将惩罚
    print("\n测试4: 长将惩罚值")
    for i in range(7):
        engine.game_state.update_check_status(True, "red")
        penalty = engine.game_state.get_perpetual_check_penalty()
        print(f"  连续{engine.game_state.consecutive_checks}次将军, 惩罚值: {penalty}")
    
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)
    

if __name__ == "__main__":
    test_perpetual_check()
