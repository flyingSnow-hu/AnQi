@echo off
REM 推荐配置 - 平衡速度和质量 (RTX 3060 12GB)

echo ========================================
echo 优化版RL AI训练 - 推荐配置
echo ========================================
echo.
echo GPU: RTX 3060 12GB
echo 配置特点:
echo - 并行6个游戏 (充分利用GPU)
echo - 批次384 (平衡显存和效率)
echo - 256通道 + 20残差块 (标准深度学习)
echo - 混合精度FP16 (2倍速度提升)
echo.
echo 预计性能:
echo - GPU利用率: 80-90%%
echo - 显存占用: 7-8GB / 12GB
echo - 训练速度: 约200-250局/小时 (4-5倍提升)
echo.
echo 按任意键开始训练...
pause

.\venv\Scripts\python.exe train_rl_ai_optimized.py ^
    --games 80 ^
    --iterations 300 ^
    --train-steps 400 ^
    --batch-size 384 ^
    --lr 0.0003 ^
    --temperature 0.7 ^
    --reward-scale 0.02 ^
    --num-channels 256 ^
    --num-res-blocks 20 ^
    --parallel-games 6 ^
    --save-interval 20 ^
    --save-dir models_v4_balanced

echo.
echo 训练完成！
pause
