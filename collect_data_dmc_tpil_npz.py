import importlib
import pathlib
import sys
import warnings
from functools import partial as bind
import os

warnings.filterwarnings('ignore', '.*box bound precision lowered.*')
warnings.filterwarnings('ignore', '.*using stateful random seeds*')
warnings.filterwarnings('ignore', '.*is a deprecated alias for.*')
warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

directory = pathlib.Path(__file__).resolve()
directory = directory.parent
sys.path.append(str(directory.parent))

from viper_rl.dreamerv3 import embodied
from viper_rl.dreamerv3.embodied import wrappers
import datetime
import random
import string

def random_uid(length=12):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def main(argv=None):
    from viper_rl.dreamerv3 import agent as agt

    parsed, other = embodied.Flags(configs=['defaults']).parse_known(argv)
    config = embodied.Config(agt.Agent.configs['defaults'])
    for name in parsed.configs:
        config = config.update(agt.Agent.configs[name])
    config = embodied.Flags(config).parse(other)
    args = embodied.Config(
        **config.run, logdir=config.logdir,
        batch_steps=config.batch_size * config.batch_length)
    print(config)

    logdir = embodied.Path(args.logdir)
    logdir.mkdirs()
    config.save(logdir / 'config.yaml')
    step = embodied.Counter()
    logger = make_logger(parsed, logdir, step, config)

    reward_model = None
    replay_kwargs = {'reward_model': reward_model}

    cleanup = []
    task_list = ['dmc_cartpole_balance']
    output_dir = './tpil_data/random_dmc/'
    os.makedirs(output_dir, exist_ok=True)

    task_episodes = 50
    import numpy as np
    import imageio.v2 as imageio

    for task in task_list:
        task_output_dir = os.path.join(output_dir, f'{task}')
        os.makedirs(task_output_dir, exist_ok=True)

        config = config.update({'task': task})
        env = make_env(config)
        task_steps = 0
        all_images = []  # For combined mp4

        for i in range(task_episodes):
            done = False
            episode_images = []
            episode_behavior = []
            while not done:
                random_action = env.env.act_space['action'].sample()
                act = {'reset': done, 'action': random_action}
                after_step = env.step(act)
                image = after_step['image']
                done = after_step['is_terminal'] or after_step['is_last']
                episode_images.append(image)
                episode_behavior.append(-1)
                all_images.append(image)  # Add to global video list
                task_steps += 1

            # Save episode as .npz
            episode_array = np.stack(episode_images)
            episode_behavior_array = np.stack(episode_behavior)
            episode_data = {
                'image': episode_array,
                'behavior': episode_behavior_array
            }
            uid1 = random_uid()
            uid2 = random_uid()
            episode_length = len(episode_images)
            timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S%f')
            filename = f'{timestamp}-{uid1}-{uid2}-{episode_length}.npz'
            out_path_npz = os.path.join(task_output_dir, filename)
            np.savez_compressed(out_path_npz, **episode_data)

            print(f'Task: {task} | Episode: {i+1}/{task_episodes} | Steps: {task_steps}')

        # Save combined MP4 of all episodes
        out_path_mp4 = os.path.join(task_output_dir, f'{task}_combined.mp4')
        imageio.mimsave(out_path_mp4, all_images, fps=20)
        print(f'Saved combined video to {out_path_mp4}')


def make_logger(parsed, logdir, step, config):
    multiplier = config.env.get(config.task.split('_')[0], {}).get('repeat', 1)
    logger = embodied.Logger(step, [
        embodied.logger.TerminalOutput(config.filter),
        embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
        embodied.logger.JSONLOutput(logdir, 'scores.jsonl', 'episode/score'),
    ], multiplier)
    return logger


def make_replay(config, directory=None, is_eval=False, rate_limit=False, reward_model=None, **kwargs):
    assert config.replay == 'uniform' or config.replay == 'uniform_relabel' or not rate_limit
    length = config.batch_length
    size = config.replay_size // 10 if is_eval else config.replay_size
    if config.replay == 'uniform_relabel':
        kw = {'online': config.replay_online}
        if rate_limit and config.run.train_ratio > 0:
            kw['samples_per_insert'] = config.run.train_ratio / config.batch_length
            kw['tolerance'] = 10 * config.batch_size
            kw['min_size'] = config.batch_size
        assert reward_model is not None, 'relabel requires reward model'
        replay = embodied.replay.UniformRelabel(
            length, reward_model, config.uniform_relabel_add_mode, size, directory, **kw)
    elif config.replay == 'uniform' or is_eval:
        kw = {'online': config.replay_online}
        if rate_limit and config.run.train_ratio > 0:
            kw['samples_per_insert'] = config.run.train_ratio / config.batch_length
            kw['tolerance'] = 10 * config.batch_size
            kw['min_size'] = config.batch_size
        replay = embodied.replay.Uniform(length, size, directory, **kw)
    elif config.replay == 'reverb':
        replay = embodied.replay.Reverb(length, size, directory)
    elif config.replay == 'chunks':
        replay = embodied.replay.NaiveChunks(length, size, directory)
    else:
        raise NotImplementedError(config.replay)
    return replay


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

