#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=7 python main.py --env-name BipedalWalker-v3 --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --use-linear-lr-decay --use-proper-time-limits --network_class MLP --n_hidden 1 --seed 10 &
CUDA_VISIBLE_DEVICES=8 python main.py --env-name BipedalWalker-v3 --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --use-linear-lr-decay --use-proper-time-limits --network_class MLP --n_hidden 1 --seed 20 &
CUDA_VISIBLE_DEVICES=7 python main.py --env-name BipedalWalker-v3 --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --use-linear-lr-decay --use-proper-time-limits --network_class MLP --n_hidden 2 --seed 10 &
CUDA_VISIBLE_DEVICES=8 python main.py --env-name BipedalWalker-v3 --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --use-linear-lr-decay --use-proper-time-limits --network_class MLP --n_hidden 2 --seed 20 &
CUDA_VISIBLE_DEVICES=7 python main.py --env-name BipedalWalker-v3 --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --use-linear-lr-decay --use-proper-time-limits --network_class FourierMLP --n_hidden 1 --sigma 0.1 --fourier_dim 256 --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=8 python main.py --env-name BipedalWalker-v3 --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --use-linear-lr-decay --use-proper-time-limits --network_class FourierMLP --n_hidden 1 --sigma 0.1 --fourier_dim 256 --train_B --seed 20 &
CUDA_VISIBLE_DEVICES=7 python main.py --env-name BipedalWalker-v3 --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --use-linear-lr-decay --use-proper-time-limits --network_class FourierMLP --n_hidden 1 --sigma 0.01 --fourier_dim 256 --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=8 python main.py --env-name BipedalWalker-v3 --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --use-linear-lr-decay --use-proper-time-limits --network_class FourierMLP --n_hidden 1 --sigma 0.01 --fourier_dim 256 --train_B --seed 20 &







8