@echo off
chcp 65001 >nul
echo ========================================
echo 强化学习AI训练 - 中等强度配置
echo ========================================
echo.
echo 配置说明:
echo   训练轮数: 200
echo   每轮局数: 30
echo   训练步数: 200
echo   批次大小: 64
echo   学习率: 0.0005
echo   探索温度: 0.7
echo.
echo 预计时间: 6-12小时
echo 预期效果: 显著提升AI棋力
echo.
pause

X:\github\xq\.venv\Scripts\python.exe train_rl_ai.py ^
  --load-model models/model_final.pth ^
  --games 30 ^
  --iterations 200 ^
  --train-steps 200 ^
  --batch-size 64 ^
  --lr 0.0005 ^
  --temperature 0.7 ^
  --save-interval 20 ^
  --save-dir models_v2

echo.
echo ========================================
echo 训练完成！
echo ========================================
pause
