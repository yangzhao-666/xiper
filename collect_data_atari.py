import importlib
import pathlib
import sys
import warnings
from functools import partial as bind
import os
import pickle
import imageio.v2 as imageio
import h5py
import numpy as np

# Suppress specific warning messages
warnings.filterwarnings('ignore', '.*box bound precision lowered.*')
warnings.filterwarnings('ignore', '.*using stateful random seeds*')
warnings.filterwarnings('ignore', '.*is a deprecated alias for.*')
warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

# Path setup
directory = pathlib.Path(__file__).resolve()
directory = directory.parent
sys.path.append(str(directory.parent))

# Import DreamerV3 framework
from viper_rl.dreamerv3 import embodied
from viper_rl.dreamerv3.embodied import wrappers

import tensorflow as tf

def main(argv=None):
    from viper_rl.dreamerv3 import agent as agt

    # Parse configuration flags
    parsed, other = embodied.Flags(configs=['defaults']).parse_known(argv)
    config = embodied.Config(agt.Agent.configs['defaults'])
    for name in parsed.configs:
        config = config.update(agt.Agent.configs[name])
    config = embodied.Flags(config).parse(other)
    args = embodied.Config(
        **config.run, logdir=config.logdir,
        batch_steps=config.batch_size * config.batch_length)
    print(config)

    # Setup logging and environment
    logdir = embodied.Path(args.logdir)
    logdir.mkdirs()
    config.save(logdir / 'config.yaml')
    step = embodied.Counter()
    logger = make_logger(parsed, logdir, step, config)

    # Load mask map (VQGAN-based) for Atari tasks
    atari_mask_path = '/home/zyang2/XViper/viper_rl_data/checkpoints/atari_vqgan/mask_map.pkl'
    mask_map = pickle.load(tf.io.gfile.GFile(atari_mask_path, 'rb'))

    # Tasks to run
    task_list = [
        'atari_pong', 
        #'atari_boxing', 
        #'atari_kangaroo'
    ]
    task_episodes = 500
    output_root = './ot_data/atari_random_color_500'
    os.makedirs(output_root, exist_ok=True)

    # Loop over tasks
    for task in task_list:
        config = config.update({'task': task})
        task_key = task[6:]  # Strip 'atari_' prefix
        mask = mask_map[task_key]
        env = make_env(config)

        task_dir = os.path.join(output_root, task)
        os.makedirs(task_dir, exist_ok=True)

        all_images = []
        task_steps = 0

        for i in range(task_episodes):
            done = False
            while not done:
                random_action = env.act_space['action'].sample()
                act = {'reset': done, 'action': random_action}
                after_step = env.step(act)
                image = after_step['image']
                image = image * mask  # Apply VQGAN mask
                done = after_step['is_terminal'] or after_step['is_last']
                all_images.append(image)
                task_steps += 1
            print(f'Task: {task} | Episode: {i+1}/{task_episodes} | Steps: {task_steps}')

        # Save outputs per task
        frames_array = np.stack(all_images)
        h5_path = os.path.join(task_dir, f'{task}_masked_images.h5')
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('frames', data=frames_array, compression='gzip')

        mp4_path = os.path.join(task_dir, f'{task}_masked_video.mp4')
        imageio.mimsave(mp4_path, all_images, fps=20)

def make_logger(parsed, logdir, step, config):
    multiplier = config.env.get(config.task.split('_')[0], {}).get('repeat', 1)
    logger = embodied.Logger(step, [
        embodied.logger.TerminalOutput(config.filter),
        embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
        embodied.logger.JSONLOutput(logdir, 'scores.jsonl', 'episode/score'),
    ], multiplier)
    return logger

def make_replay(config, directory=None, is_eval=False, rate_limit=False, reward_model=None, **kwargs):
    assert config.replay in ['uniform', 'uniform_relabel', 'reverb', 'chunks'] or not rate_limit
    length = config.batch_length
    size = config.replay_size // 10 if is_eval else config.replay_size
    if config.replay == 'uniform_relabel':
        assert reward_model is not None, 'relabel requires reward model'
        kw = {'online': config.replay_online}
        if rate_limit and config.run.train_ratio > 0:
            kw.update({
                'samples_per_insert': config.run.train_ratio / config.batch_length,
                'tolerance': 10 * config.batch_size,
                'min_size': config.batch_size
            })
        return embodied.replay.UniformRelabel(length, reward_model, config.uniform_relabel_add_mode, size, directory, **kw)
    elif config.replay == 'uniform' or is_eval:
        kw = {'online': config.replay_online}
        if rate_limit and config.run.train_ratio > 0:
            kw.update({
                'samples_per_insert': config.run.train_ratio / config.batch_length,
                'tolerance': 10 * config.batch_size,
                'min_size': config.batch_size
            })
        return embodied.replay.Uniform(length, size, directory, **kw)
    elif config.replay == 'reverb':
        return embodied.replay.Reverb(length, size, directory)
    elif config.replay == 'chunks':
        return embodied.replay.NaiveChunks(length, size, directory)
    else:
        raise NotImplementedError(config.replay)

def make_envs(config, **overrides):
    suite, task = config.task.split('_', 1)
    ctors = []
    for index in range(config.envs.amount):
        ctor = lambda: make_env(config, **overrides)
        if config.envs.parallel != 'none':
            ctor = bind(embodied.Parallel, ctor, config.envs.parallel)
        if config.envs.restart:
            ctor = bind(wrappers.RestartOnException, ctor)
        ctors.append(ctor)
    envs = [ctor() for ctor in ctors]
    return embodied.BatchEnv(envs, parallel=(config.envs.parallel != 'none'))

def make_env(config, **overrides):
    suite, task = config.task.split('_', 1)
    ctor = {
        'dummy': 'embodied.envs.dummy:Dummy',
        'gym': 'embodied.envs.from_gym:FromGym',
        'dm': 'embodied.envs.from_dmenv:FromDM',
        'crafter': 'embodied.envs.crafter:Crafter',
        'dmc': 'embodied.envs.dmc:DMC',
        'rlbench': 'embodied.envs.rlbench:RLBench',
        'dmcmulticam': 'embodied.envs.dmcmulticam:DMCMultiCam',
        'atari': 'embodied.envs.atari:Atari',
        'dmlab': 'embodied.envs.dmlab:DMLab',
        'minecraft': 'embodied.envs.minecraft:Minecraft',
        'loconav': 'embodied.envs.loconav:LocoNav',
        'pinpad': 'embodied.envs.pinpad:PinPad',
        'kitchen': 'embodied.envs.kitchen:Kitchen',
        'cliport': 'embodied.envs.cliport:Cliport',
    }[suite]
    if isinstance(ctor, str):
        module, cls = ctor.split(':')
        module = importlib.import_module(module)
        ctor = getattr(module, cls)
    kwargs = config.env.get(suite, {})
    kwargs.update(overrides)
    env = ctor(task, **kwargs)
    return wrap_env(env, config)

def wrap_env(env, config):
    args = config.wrapper
    for name, space in env.act_space.items():
        if name == 'reset':
            continue
        elif space.discrete:
            env = wrappers.OneHotAction(env, name)
        elif args.discretize:
            env = wrappers.DiscretizeAction(env, name, args.discretize)
        else:
            env = wrappers.NormalizeAction(env, name)
    if args.density:
        env = wrappers.Density(env)
    env = wrappers.FlattenTwoDimObs(env)
    env = wrappers.ExpandScalars(env)
    if args.length:
        env = wrappers.TimeLimit(env, args.length, args.reset)
    if args.checks:
        env = wrappers.CheckSpaces(env)
    for name, space in env.act_space.items():
        if not space.discrete:
            env = wrappers.ClipAction(env, name)
    return env

if __name__ == '__main__':
    main()

