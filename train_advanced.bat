@echo off
chcp 65001 >nul
echo ========================================
echo 强化学习AI训练 - 深度训练配置
echo ========================================
echo.
echo 配置说明:
echo   训练轮数: 500
echo   每轮局数: 50
echo   训练步数: 300
echo   批次大小: 128
echo   学习率: 0.0003
echo   探索温度: 0.6
echo.
echo 预计时间: 24-48小时
echo 预期效果: 高水平AI棋力
echo.
echo ⚠️ 警告: 这是长时间训练任务！
echo.
pause

X:\github\xq\.venv\Scripts\python.exe train_rl_ai.py ^
  --load-model models/model_final.pth ^
  --games 50 ^
  --iterations 500 ^
  --train-steps 300 ^
  --batch-size 128 ^
  --lr 0.0003 ^
  --temperature 0.6 ^
  --save-interval 25 ^
  --save-dir models_advanced

echo.
echo ========================================
echo 训练完成！
echo ========================================
pause
