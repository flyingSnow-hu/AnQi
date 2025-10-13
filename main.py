#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
揭棋游戏主程序
支持人人对战，预留人机对战接口
"""

import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from ui.dark_chess_gui import DarkChessGUI

def main():
    """主函数"""
    try:
        game = DarkChessGUI()
        game.run()
    except KeyboardInterrupt:
        print("\n游戏被用户中断")
    except Exception as e:
        print(f"游戏运行时出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()