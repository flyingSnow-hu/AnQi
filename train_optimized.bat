@echo off
REM 优化版训练脚本 - 最大化GPU利用率
REM 使用更大批次、并行自对弈、混合精度训练

echo ========================================
echo 优化版RL AI训练 - 高性能配置
echo ========================================
echo.
echo 优化特性:
echo - 并行8个自对弈游戏
echo - 批次大小512 (充分利用GPU内存)
echo - 混合精度训练 (FP16)
echo - 256通道 + 20残差块网络
echo.
echo 按任意键开始训练...
pause

venv\Scripts\python.exe train_rl_ai_optimized.py ^
    --games 100 ^
    --iterations 500 ^
    --train-steps 500 ^
    --batch-size 512 ^
    --lr 0.0003 ^
    --temperature 0.7 ^
    --reward-scale 0.02 ^
    --num-channels 256 ^
    --num-res-blocks 20 ^
    --parallel-games 8 ^
    --save-interval 20 ^
    --save-dir models_v4_optimized

echo.
echo 训练完成！
pause
