total = 0
envs = ['Ant-v2', 'Humanoid-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Walker2d-v2']
for env in envs:
    commands = []
    base_command = f'python main.py --env-name {env} --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 1000000 --use-linear-lr-decay --use-proper-time-limits --hidden_dim 256'
    for n_hidden in [1,2]:
        commands.append(f'{base_command} --network_class MLP --n_hidden {n_hidden}')

    for type in ['--concatenate_fourier --train_B']:
        for n_hidden in [1, 2]:
            for fourier_dim in [64, 256, 1024]:
                for sigma in [0.01]:
                    commands.append(f'{base_command} --network_class FourierMLP --n_hidden {n_hidden} --sigma {sigma} --fourier_dim {fourier_dim} {type}')
    count = 0
    for command in commands:
        gpus = list(range(10))
        for seed in [10]:
            if total % 8 == 0:
                print(total)
            total += 1
            print(f'CUDA_VISIBLE_DEVICES={gpus[total % len(gpus)]} {command} --seed {seed} &')
            count = (count + 1) % len(gpus)