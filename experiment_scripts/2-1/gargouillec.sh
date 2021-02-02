#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=6 python main.py --env-name BipedalWalkerHardcore-v3 --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 2.5e-4 --entropy-coef 0.001 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --use-proper-time-limits --network_class MLP --n_hidden 1 --seed 10 &
CUDA_VISIBLE_DEVICES=7 python main.py --env-name BipedalWalkerHardcore-v3 --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 2.5e-4 --entropy-coef 0.001 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --use-proper-time-limits --network_class MLP --n_hidden 1 --seed 20 &
CUDA_VISIBLE_DEVICES=8 python main.py --env-name BipedalWalkerHardcore-v3 --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 2.5e-4 --entropy-coef 0.001 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --use-proper-time-limits --network_class MLP --n_hidden 2 --seed 10 &
CUDA_VISIBLE_DEVICES=9 python main.py --env-name BipedalWalkerHardcore-v3 --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 2.5e-4 --entropy-coef 0.001 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --use-proper-time-limits --network_class MLP --n_hidden 2 --seed 20 &
CUDA_VISIBLE_DEVICES=6 python main.py --env-name BipedalWalkerHardcore-v3 --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 2.5e-4 --entropy-coef 0.001 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --use-proper-time-limits --network_class FourierMLP --n_hidden 1 --sigma 0.03 --fourier_dim 256 --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=7 python main.py --env-name BipedalWalkerHardcore-v3 --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 2.5e-4 --entropy-coef 0.001 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --use-proper-time-limits --network_class FourierMLP --n_hidden 1 --sigma 0.03 --fourier_dim 256 --train_B --seed 20 &
CUDA_VISIBLE_DEVICES=8 python main.py --env-name BipedalWalkerHardcore-v3 --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 2.5e-4 --entropy-coef 0.001 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --use-proper-time-limits --network_class FourierMLP --n_hidden 1 --sigma 0.01 --fourier_dim 256 --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=9 python main.py --env-name BipedalWalkerHardcore-v3 --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 2.5e-4 --entropy-coef 0.001 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --use-proper-time-limits --network_class FourierMLP --n_hidden 1 --sigma 0.01 --fourier_dim 256 --train_B --seed 20 &
CUDA_VISIBLE_DEVICES=6 python main.py --env-name BipedalWalkerHardcore-v3 --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 2.5e-4 --entropy-coef 0.001 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --use-proper-time-limits --network_class FourierMLP --n_hidden 1 --sigma 0.003 --fourier_dim 256 --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=7 python main.py --env-name BipedalWalkerHardcore-v3 --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 2.5e-4 --entropy-coef 0.001 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --use-proper-time-limits --network_class FourierMLP --n_hidden 1 --sigma 0.003 --fourier_dim 256 --train_B --seed 20 &
CUDA_VISIBLE_DEVICES=8 python main.py --env-name BipedalWalkerHardcore-v3 --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 2.5e-4 --entropy-coef 0.001 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --use-proper-time-limits --network_class FourierMLP --n_hidden 1 --sigma 0.03 --fourier_dim 256 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=9 python main.py --env-name BipedalWalkerHardcore-v3 --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 2.5e-4 --entropy-coef 0.001 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --use-proper-time-limits --network_class FourierMLP --n_hidden 1 --sigma 0.03 --fourier_dim 256 --concatenate_fourier --train_B --seed 20 &
CUDA_VISIBLE_DEVICES=6 python main.py --env-name BipedalWalkerHardcore-v3 --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 2.5e-4 --entropy-coef 0.001 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --use-proper-time-limits --network_class FourierMLP --n_hidden 1 --sigma 0.01 --fourier_dim 256 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=7 python main.py --env-name BipedalWalkerHardcore-v3 --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 2.5e-4 --entropy-coef 0.001 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --use-proper-time-limits --network_class FourierMLP --n_hidden 1 --sigma 0.01 --fourier_dim 256 --concatenate_fourier --train_B --seed 20 &
CUDA_VISIBLE_DEVICES=8 python main.py --env-name BipedalWalkerHardcore-v3 --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 2.5e-4 --entropy-coef 0.001 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --use-proper-time-limits --network_class FourierMLP --n_hidden 1 --sigma 0.003 --fourier_dim 256 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=9 python main.py --env-name BipedalWalkerHardcore-v3 --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 2.5e-4 --entropy-coef 0.001 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --use-proper-time-limits --network_class FourierMLP --n_hidden 1 --sigma 0.003 --fourier_dim 256 --concatenate_fourier --train_B --seed 20 &
