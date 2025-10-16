import tkinter as tk
from tkinter import messagebox
from typing import Optional, Tuple
from game.game_engine import GameEngine
from players.human_player import HumanPlayer

class DarkChessGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("揭棋游戏")
        self.root.geometry("1000x700")
        self.root.resizable(False, False)
        
        self.game_engine = None
        self.selected_pos = None
        self.valid_moves = []
        
        # AI相关
        self.ai_thinking = False
        self.game_mode = None  # 游戏模式
        
        # 界面参数
        self.cell_size = 60
        self.board_start_x = 50
        self.board_start_y = 50
        
        self.setup_ui()
        
    def setup_ui(self):
        """设置主菜单"""
        self.menu_frame = tk.Frame(self.root, bg='lightgray')
        self.menu_frame.pack(expand=True, fill='both')
        
        tk.Label(self.menu_frame, text="揭棋游戏", 
                font=("SimHei", 24, "bold"), bg='lightgray').pack(pady=20)
        
        tk.Label(self.menu_frame, text="选择游戏模式", 
                font=("SimHei", 16), bg='lightgray').pack(pady=10)
        
        # 人类对战
        tk.Button(self.menu_frame, text="人类 vs 人类", 
                 font=("SimHei", 13), width=25, height=2,
                 bg='#4CAF50', fg='white',
                 command=lambda: self.start_game("human_vs_human")).pack(pady=5)
        
        # 人机对战 - 红方人类
        tk.Button(self.menu_frame, text="人类 vs AI简单 (红方人类)", 
                 font=("SimHei", 12), width=25, height=1,
                 bg='#2196F3', fg='white',
                 command=lambda: self.start_game("human_vs_ai_red_easy")).pack(pady=3)
        
        tk.Button(self.menu_frame, text="人类 vs AI困难 (红方人类)", 
                 font=("SimHei", 12), width=25, height=1,
                 bg='#1976D2', fg='white',
                 command=lambda: self.start_game("human_vs_ai_red_hard")).pack(pady=3)
        
        # 人机对战 - 黑方人类
        tk.Button(self.menu_frame, text="人类 vs AI简单 (黑方人类)", 
                 font=("SimHei", 12), width=25, height=1,
                 bg='#2196F3', fg='white',
                 command=lambda: self.start_game("human_vs_ai_black_easy")).pack(pady=3)
        
        tk.Button(self.menu_frame, text="人类 vs AI困难 (黑方人类)", 
                 font=("SimHei", 12), width=25, height=1,
                 bg='#1976D2', fg='white',
                 command=lambda: self.start_game("human_vs_ai_black_hard")).pack(pady=3)
        
        # RL AI对战
        tk.Label(self.menu_frame, text="强化学习AI", 
                font=("SimHei", 14), bg='lightgray', fg='purple').pack(pady=5)
        
        tk.Button(self.menu_frame, text="人类 vs RL AI (红方人类)", 
                 font=("SimHei", 12), width=25, height=1,
                 bg='#9C27B0', fg='white',
                 command=lambda: self.start_game("human_vs_rl_red")).pack(pady=3)
        
        tk.Button(self.menu_frame, text="人类 vs RL AI (黑方人类)", 
                 font=("SimHei", 12), width=25, height=1,
                 bg='#9C27B0', fg='white',
                 command=lambda: self.start_game("human_vs_rl_black")).pack(pady=3)
        
        tk.Button(self.menu_frame, text="MCTS AI vs RL AI", 
                 font=("SimHei", 12), width=25, height=1,
                 bg='#7B1FA2', fg='white',
                 command=lambda: self.start_game("mcts_vs_rl")).pack(pady=3)
        
        # AI对战
        tk.Label(self.menu_frame, text="AI对战观战模式", 
                font=("SimHei", 14), bg='lightgray').pack(pady=5)
        
        tk.Button(self.menu_frame, text="AI简单 vs AI简单", 
                 font=("SimHei", 12), width=25, height=1,
                 bg='#FF9800', fg='white',
                 command=lambda: self.start_game("ai_easy_vs_ai_easy")).pack(pady=3)
        
        tk.Button(self.menu_frame, text="AI简单 vs AI困难", 
                 font=("SimHei", 12), width=25, height=1,
                 bg='#FF9800', fg='white',
                 command=lambda: self.start_game("ai_vs_ai")).pack(pady=3)
        
        tk.Button(self.menu_frame, text="AI困难 vs AI困难", 
                 font=("SimHei", 12), width=25, height=1,
                 bg='#E65100', fg='white',
                 command=lambda: self.start_game("ai_hard_vs_ai_hard")).pack(pady=3)
        
        # 游戏规则简介
        rules_text = """
游戏规则：
• 将帅固定在原位并翻开，其他棋子随机摆放
• 所有非将帅棋子初始背面朝上
• 移动时自动翻开显示真实兵种
• 炮可以吃对方任何棋子（包括暗棋）
• 将帅面对面无遮挡时可以飞将吃掉对方
• 吃掉对方将帅获胜

AI类型：
• 简单AI：快速启发式算法
• 困难AI：MCTS算法（2000次模拟）
• RL AI：深度强化学习（神经网络）
        """
        tk.Label(self.menu_frame, text=rules_text, 
                font=("SimHei", 9), bg='lightgray', 
                justify='left').pack(pady=5)
        
        self.game_frame = tk.Frame(self.root)
        
    def start_game(self, mode):
        """开始游戏"""
        self.game_mode = mode
        self.menu_frame.pack_forget()
        self.game_frame.pack(expand=True, fill='both')
        
        red_player = HumanPlayer("红方玩家", "red")
        black_player = HumanPlayer("黑方玩家", "black")
            
        self.game_engine = GameEngine(red_player, black_player)
        self.game_engine.add_observer(self)
        
        # 根据模式设置AI
        if mode == "human_vs_ai_red_easy":
            # 红方人类，黑方简单AI
            from ai.mcts_ai import SimpleAI
            ai = SimpleAI("black")
            self.game_engine.set_ai_player("black", ai)
        elif mode == "human_vs_ai_red_hard":
            # 红方人类，黑方困难AI
            from ai.mcts_ai import MCTSAI
            ai = MCTSAI("black", simulations=1000)
            self.game_engine.set_ai_player("black", ai)
        elif mode == "human_vs_ai_black_easy":
            # 黑方人类，红方简单AI
            from ai.mcts_ai import SimpleAI
            ai = SimpleAI("red")
            self.game_engine.set_ai_player("red", ai)
        elif mode == "human_vs_ai_black_hard":
            # 黑方人类，红方困难AI
            from ai.mcts_ai import MCTSAI
            ai = MCTSAI("red", simulations=1000)
            self.game_engine.set_ai_player("red", ai)
        elif mode == "ai_vs_ai":
            # 双方都是AI（简单 vs 困难）
            from ai.mcts_ai import SimpleAI, MCTSAI
            self.game_engine.set_ai_player("red", SimpleAI("red"))
            self.game_engine.set_ai_player("black", MCTSAI("black", simulations=1000))
        elif mode == "ai_easy_vs_ai_easy":
            # 双方都是简单AI
            from ai.mcts_ai import SimpleAI
            self.game_engine.set_ai_player("red", SimpleAI("red"))
            self.game_engine.set_ai_player("black", SimpleAI("black"))
        elif mode == "ai_hard_vs_ai_hard":
            # 双方都是困难AI
            from ai.mcts_ai import MCTSAI
            self.game_engine.set_ai_player("red", MCTSAI("red", simulations=1000))
            self.game_engine.set_ai_player("black", MCTSAI("black", simulations=1000))
        elif mode == "human_vs_rl_red":
            # 红方人类，黑方RL AI
            from rl_ai.rl_player import RLPlayer
            import os
            model_path = "models_v3/model_final.pth"
            if not os.path.exists(model_path):
                messagebox.showwarning("警告", "未找到训练好的模型！将使用随机初始化模型")
                model_path = None
            ai = RLPlayer("black", model_path=model_path, temperature=0.0)
            self.game_engine.set_ai_player("black", ai)
        elif mode == "human_vs_rl_black":
            # 黑方人类，红方RL AI
            from rl_ai.rl_player import RLPlayer
            import os
            model_path = "models_v3/model_final.pth"
            if not os.path.exists(model_path):
                messagebox.showwarning("警告", "未找到训练好的模型！将使用随机初始化模型")
                model_path = None
            ai = RLPlayer("red", model_path=model_path, temperature=0.0)
            self.game_engine.set_ai_player("red", ai)
        elif mode == "mcts_vs_rl":
            # MCTS vs RL AI对战
            from ai.mcts_ai import MCTSAI
            from rl_ai.rl_player import RLPlayer
            import os
            model_path = "models_v3/model_final.pth"
            if not os.path.exists(model_path):
                messagebox.showwarning("警告", "未找到训练好的模型！将使用随机初始化模型")
                model_path = None
            self.game_engine.set_ai_player("red", MCTSAI("red", simulations=2000))
            self.game_engine.set_ai_player("black", RLPlayer("black", model_path=model_path, temperature=0.0))
        
        self.setup_game_board()
        
        # 如果是AI先手，启动AI思考
        if self.game_engine.is_ai_turn():
            self.root.after(1000, self.ai_make_move)
        
    def setup_game_board(self):
        """设置游戏棋盘"""
        # 清空game_frame
        for widget in self.game_frame.winfo_children():
            widget.destroy()
            
        # 左侧棋盘
        board_frame = tk.Frame(self.game_frame)
        board_frame.pack(side=tk.LEFT, padx=20, pady=20)
        
        canvas_width = self.board_start_x * 2 + 8 * self.cell_size
        canvas_height = self.board_start_y * 2 + 9 * self.cell_size
        
        self.canvas = tk.Canvas(board_frame, width=canvas_width, height=canvas_height, 
                               bg='burlywood', highlightthickness=2, 
                               highlightbackground='black')
        self.canvas.pack()
        
        # 右侧信息面板
        info_frame = tk.Frame(self.game_frame, width=300)
        info_frame.pack(side=tk.RIGHT, fill='y', padx=20, pady=20)
        info_frame.pack_propagate(False)
        
        # 游戏状态
        status_frame = tk.LabelFrame(info_frame, text="游戏状态", font=("SimHei", 12))
        status_frame.pack(fill='x', pady=10)
        
        self.status_label = tk.Label(status_frame, text="红方先行", 
                                    font=("SimHei", 14), fg='red')
        self.status_label.pack(pady=10)
        
        self.move_count_label = tk.Label(status_frame, text="回合数: 0", 
                                        font=("SimHei", 10))
        self.move_count_label.pack(pady=5)
        
        # 被吃棋子
        captured_frame = tk.LabelFrame(info_frame, text="被吃棋子", font=("SimHei", 12))
        captured_frame.pack(fill='x', pady=10)
        
        self.red_captured_label = tk.Label(captured_frame, text="红方被吃: ", 
                                          font=("SimHei", 10), fg='red')
        self.red_captured_label.pack(anchor='w', padx=5)
        
        self.black_captured_label = tk.Label(captured_frame, text="黑方被吃: ", 
                                            font=("SimHei", 10), fg='black')
        self.black_captured_label.pack(anchor='w', padx=5)
        
        # 操作按钮
        button_frame = tk.LabelFrame(info_frame, text="操作", font=("SimHei", 12))
        button_frame.pack(fill='x', pady=10)
        
        tk.Button(button_frame, text="认输", font=("SimHei", 12),
                 command=self.surrender).pack(pady=5, fill='x')
        
        tk.Button(button_frame, text="返回主菜单", font=("SimHei", 12),
                 command=self.back_to_menu).pack(pady=5, fill='x')
        
        # 移动历史
        history_frame = tk.LabelFrame(info_frame, text="移动历史", font=("SimHei", 12))
        history_frame.pack(fill='both', expand=True, pady=10)
        
        history_text_frame = tk.Frame(history_frame)
        history_text_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.history_text = tk.Text(history_text_frame, font=("SimHei", 9), 
                                   state='disabled', wrap='word')
        history_scrollbar = tk.Scrollbar(history_text_frame, orient='vertical', 
                                        command=self.history_text.yview)
        self.history_text.configure(yscrollcommand=history_scrollbar.set)
        
        self.history_text.pack(side='left', fill='both', expand=True)
        history_scrollbar.pack(side='right', fill='y')
        
        # 绑定鼠标事件
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        
        # 绘制棋盘和棋子
        self.draw_board()
        self.draw_pieces()
        
    def draw_board(self):
        """绘制棋盘"""
        self.canvas.delete("board")
        
        # 横线
        for i in range(10):
            y = self.board_start_y + i * self.cell_size
            self.canvas.create_line(self.board_start_x, y, 
                                   self.board_start_x + 8 * self.cell_size, y, 
                                   width=2, tags="board")
        
        # 竖线
        for i in range(9):
            x = self.board_start_x + i * self.cell_size
            self.canvas.create_line(x, self.board_start_y, 
                                   x, self.board_start_y + 9 * self.cell_size, 
                                   width=2, tags="board")
        
        # 楚河汉界
        river_y = self.board_start_y + 4.5 * self.cell_size
        self.canvas.create_text(self.board_start_x + 2 * self.cell_size, river_y, 
                               text="楚河", font=("SimHei", 14), tags="board")
        self.canvas.create_text(self.board_start_x + 6 * self.cell_size, river_y, 
                               text="汉界", font=("SimHei", 14), tags="board")
        
        # 九宫格斜线
        palace_lines = [(3, 0, 5, 2), (5, 0, 3, 2), (3, 7, 5, 9), (5, 7, 3, 9)]
        for x1, y1, x2, y2 in palace_lines:
            self.canvas.create_line(
                self.board_start_x + x1 * self.cell_size,
                self.board_start_y + y1 * self.cell_size,
                self.board_start_x + x2 * self.cell_size,
                self.board_start_y + y2 * self.cell_size,
                width=2, tags="board"
            )
            
    def draw_pieces(self):
        """绘制棋子"""
        self.canvas.delete("piece")
        
        board = self.game_engine.board
        
        for row in range(10):
            for col in range(9):
                piece = board.get_piece(row, col)
                if piece:
                    x = self.board_start_x + col * self.cell_size
                    y = self.board_start_y + row * self.cell_size
                    
                    if piece.is_revealed():
                        fill_color = "lightyellow"
                        text_color = "red" if piece.color == "red" else "black"
                        text = piece.symbol
                    else:
                        fill_color = "lightgray"
                        text_color = "red" if piece.color == "red" else "black"
                        text = piece.symbol  # ○ 或 ●
                    
                    radius = self.cell_size // 3
                    self.canvas.create_oval(x - radius, y - radius, 
                                          x + radius, y + radius,
                                          fill=fill_color, outline=text_color, 
                                          width=2, tags="piece")
                    
                    self.canvas.create_text(x, y, text=text, 
                                          font=("SimHei", 14, "bold"), 
                                          fill=text_color, tags="piece")
                                          
    def on_canvas_click(self, event):
        """处理画布点击"""
        if self.game_engine.game_state.game_over:
            return
            
        col = round((event.x - self.board_start_x) / self.cell_size)
        row = round((event.y - self.board_start_y) / self.cell_size)
        
        if 0 <= row < 10 and 0 <= col < 9:
            self.handle_click(row, col)
            
    def handle_click(self, row: int, col: int):
        """处理棋子点击逻辑"""
        if self.selected_pos is None:
            # 选择棋子
            piece = self.game_engine.board.get_piece(row, col)
            if piece and piece.color == self.game_engine.game_state.current_turn:
                self.selected_pos = (row, col)
                self.valid_moves = self.game_engine.get_valid_moves_for_position((row, col))
                self.highlight_selection()
        else:
            # 尝试移动
            to_pos = (row, col)
            
            if to_pos == self.selected_pos:
                self.clear_selection()
            elif to_pos in self.valid_moves:
                success = self.game_engine.make_move(self.selected_pos, to_pos)
                if success:
                    self.clear_selection()
                    self.draw_pieces()
            else:
                # 重新选择
                piece = self.game_engine.board.get_piece(row, col)
                if piece and piece.color == self.game_engine.game_state.current_turn:
                    self.selected_pos = (row, col)
                    self.valid_moves = self.game_engine.get_valid_moves_for_position((row, col))
                    self.highlight_selection()
                else:
                    self.clear_selection()
                    
    def highlight_selection(self):
        """高亮显示选中和可移动位置"""
        self.canvas.delete("highlight")
        
        if self.selected_pos:
            row, col = self.selected_pos
            x = self.board_start_x + col * self.cell_size
            y = self.board_start_y + row * self.cell_size
            radius = self.cell_size // 3 + 5
            
            self.canvas.create_oval(x - radius, y - radius, 
                                  x + radius, y + radius,
                                  outline="blue", width=3, tags="highlight")
            
            for move_row, move_col in self.valid_moves:
                move_x = self.board_start_x + move_col * self.cell_size
                move_y = self.board_start_y + move_row * self.cell_size
                
                self.canvas.create_oval(move_x - 8, move_y - 8, 
                                      move_x + 8, move_y + 8,
                                      fill="lightgreen", outline="green", 
                                      width=2, tags="highlight")
                                      
    def clear_selection(self):
        """清除选择"""
        self.selected_pos = None
        self.valid_moves = []
        self.canvas.delete("highlight")
        
    def update_status_display(self):
        """更新状态显示"""
        if self.game_engine.game_state.game_over:
            self.status_label.config(text=f"游戏结束 - {self.game_engine.game_state.winner}获胜!")
        else:
            current_color = self.game_engine.game_state.current_turn
            color_text = "红方" if current_color == "red" else "黑方"
            self.status_label.config(text=f"{color_text}行棋", fg=current_color)
            
        self.move_count_label.config(text=f"回合数: {self.game_engine.game_state.move_count}")
        
        # 使用棋子的中文名称显示被吃棋子
        red_captured = [p.symbol for p in self.game_engine.game_state.captured_pieces["red"]]
        black_captured = [p.symbol for p in self.game_engine.game_state.captured_pieces["black"]]
        
        self.red_captured_label.config(text=f"红方被吃: {' '.join(red_captured)}")
        self.black_captured_label.config(text=f"黑方被吃: {' '.join(black_captured)}")
        
    def add_move_to_history(self, move_record):
        """添加移动到历史记录"""
        self.history_text.config(state='normal')
        
        # 获取移动的棋子对象以显示中文名称
        piece = self.game_engine.board.get_piece(*move_record['to'])
        piece_name = piece.symbol if piece else move_record['piece']
        
        # 颜色转换为中文
        color_text = "红" if move_record['piece_color'] == "red" else "黑"
        
        # 格式化坐标显示（行，列）
        from_pos = move_record['from']
        to_pos = move_record['to']
        pos_text = f"({from_pos[0]},{from_pos[1]})→({to_pos[0]},{to_pos[1]})"
        
        move_text = f"{move_record['move_number']}. "
        if move_record['was_hidden']:
            move_text += f"{color_text}方翻开{piece_name} {pos_text}"
        else:
            move_text += f"{color_text}方{piece_name} {pos_text}"
            
        if move_record['captured']:
            # 获取被吃棋子的中文名称
            captured_color = "红" if move_record['captured_color'] == "red" else "黑"
            # 从被吃棋子列表中获取刚被吃的棋子
            captured_pieces = self.game_engine.game_state.captured_pieces[move_record['captured_color']]
            if captured_pieces:
                captured_name = captured_pieces[-1].symbol
                move_text += f" 吃 {captured_color}方{captured_name}"
            
        move_text += "\n"
        
        self.history_text.insert(tk.END, move_text)
        self.history_text.see(tk.END)
        self.history_text.config(state='disabled')
        
    def on_game_event(self, event: str, data):
        """处理游戏事件"""
        if event == "move_made":
            self.add_move_to_history(data)
            self.update_status_display()
            # AI移动后，检查是否需要继续AI思考
            if self.game_engine.is_ai_turn() and not self.game_engine.game_state.game_over:
                self.root.after(500, self.ai_make_move)
        elif event == "turn_changed":
            self.update_status_display()
            # 回合切换后，检查是否需要AI思考
            if self.game_engine.is_ai_turn() and not self.game_engine.game_state.game_over:
                self.root.after(500, self.ai_make_move)
        elif event == "game_over":
            self.update_status_display()
            messagebox.showinfo("游戏结束", f"{data['winner']}获胜!\n原因: {data['reason']}")
    
    def ai_make_move(self):
        """AI执行移动"""
        if self.ai_thinking or self.game_engine.game_state.game_over:
            return
        
        if not self.game_engine.is_ai_turn():
            return
        
        self.ai_thinking = True
        current_color = self.game_engine.game_state.current_turn
        color_text = "红方" if current_color == "red" else "黑方"
        self.status_label.config(text=f"{color_text}AI思考中...")
        
        # 在后台线程中计算AI移动
        import threading
        def think():
            move = self.game_engine.get_ai_move()
            self.root.after(0, lambda: self.execute_ai_move(move))
        
        thread = threading.Thread(target=think)
        thread.daemon = True
        thread.start()
    
    def execute_ai_move(self, move):
        """执行AI计算出的移动"""
        self.ai_thinking = False
        
        if move and not self.game_engine.game_state.game_over:
            from_pos, to_pos = move
            if self.game_engine.make_move(from_pos, to_pos):
                self.draw_pieces()
                self.update_status_display()
                
    def surrender(self):
        """认输"""
        if not self.game_engine or self.game_engine.game_state.game_over:
            return
            
        current_player = "红方" if self.game_engine.game_state.current_turn == "red" else "黑方"
        result = messagebox.askyesno("认输", f"{current_player}确定要认输吗？")
        if result:
            winner = "黑方" if self.game_engine.game_state.current_turn == "red" else "红方"
            self.game_engine.game_state.end_game(winner)
            self.update_status_display()
            messagebox.showinfo("游戏结束", f"{winner}获胜！\n{current_player}认输")
            
    def back_to_menu(self):
        """返回主菜单"""
        self.clear_selection()
        self.game_frame.pack_forget()
        self.menu_frame.pack(expand=True, fill='both')
        self.game_engine = None
        
    def run(self):
        """运行游戏"""
        self.root.mainloop()