total = 0
envs = ['BipedalWalker-v3', 'CarRacing-v0', 'LunarLanderContinuous-v2', 'BipedalWalkerHardcore-v3']
for env in envs:
    commands = []
    base_command = f'python main.py --env-name {env} --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --use-linear-lr-decay --use-proper-time-limits'
    for n_hidden in [1,2]:
        commands.append(f'{base_command} --network_class MLP --n_hidden {n_hidden}')

    for type in ['--train_B', '--concatenate_fourier --train_B']:
        for n_hidden in [1]:
            for fourier_dim in [256]:
                for sigma in [0.01, 0.001]:
                    commands.append(f'{base_command} --network_class FourierMLP --n_hidden {n_hidden} --sigma {sigma} --fourier_dim {fourier_dim}')
    count = 0
    for command in commands:
        # gpus = list(range(8,10))
        gpus = list(range(10))
        for seed in [10, 20]:
            if total % 8 == 0:
                print(total)
            total += 1
            print(f'CUDA_VISIBLE_DEVICES={gpus[count]} {command} --seed {seed} &')
            count = (count + 1) % len(gpus)