#!/usr/bin/env bash

# humanoid
CUDA_VISIBLE_DEVICES=7 python main.py --env-name Humanoid-v2 --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 1000000 --use-linear-lr-decay --use-proper-time-limits --hidden_dim 256 --network_class MLP --n_hidden 1 --seed 20 &
CUDA_VISIBLE_DEVICES=8 python main.py --env-name Humanoid-v2 --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 1000000 --use-linear-lr-decay --use-proper-time-limits --hidden_dim 256 --network_class FourierMLP --n_hidden 1 --sigma 0.01 --fourier_dim 256 --concatenate_fourier --train_B --seed 20 &

# half cheetah
CUDA_VISIBLE_DEVICES=9 python main.py --env-name HalfCheetah-v2 --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 1000000 --use-linear-lr-decay --use-proper-time-limits --hidden_dim 256 --network_class MLP --n_hidden 1 --seed 20 &
CUDA_VISIBLE_DEVICES=7 python main.py --env-name HalfCheetah-v2 --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 1000000 --use-linear-lr-decay --use-proper-time-limits --hidden_dim 256 --network_class FourierMLP --n_hidden 1 --sigma 0.01 --fourier_dim 64 --concatenate_fourier --train_B --seed 20 &

# hopper
CUDA_VISIBLE_DEVICES=8 python main.py --env-name Hopper-v2 --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 1000000 --use-linear-lr-decay --use-proper-time-limits --hidden_dim 256 --network_class MLP --n_hidden 1 --seed 20 &
CUDA_VISIBLE_DEVICES=9 python main.py --env-name Hopper-v2 --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 1000000 --use-linear-lr-decay --use-proper-time-limits --hidden_dim 256 --network_class FourierMLP --n_hidden 1 --sigma 0.01 --fourier_dim 64 --concatenate_fourier --train_B --seed 20 &

# walker
CUDA_VISIBLE_DEVICES=7 python main.py --env-name Walker2d-v2 --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 1000000 --use-linear-lr-decay --use-proper-time-limits --hidden_dim 256 --network_class MLP --n_hidden 1 --seed 20 &
CUDA_VISIBLE_DEVICES=8 python main.py --env-name Walker2d-v2 --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 1000000 --use-linear-lr-decay --use-proper-time-limits --hidden_dim 256 --network_class FourierMLP --n_hidden 1 --sigma 0.01 --fourier_dim 64 --concatenate_fourier --train_B --seed 20 &
CUDA_VISIBLE_DEVICES=9 python main.py --env-name Walker2d-v2 --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 1000000 --use-linear-lr-decay --use-proper-time-limits --hidden_dim 256 --network_class FourierMLP --n_hidden 1 --sigma 0.01 --fourier_dim 256 --concatenate_fourier --train_B --seed 20 &



