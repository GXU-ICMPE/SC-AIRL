import os
import argparse
import torch
import gym
import numpy as np
from pybullet_envs.gym_locomotion_envs import AntBulletEnv, HopperBulletEnv

from gail_airl_ppo.env import make_env
from gail_airl_ppo.algo import SACExpert
from gail_airl_ppo.algo.ppo import PPOExpert
from gail_airl_ppo.utils import collect_demo
from gym.envs.classic_control.ur5Env_1 import ur5Env_1
from env.panda_grasp_stack import ur5Env_1

# from realur5 import Robot


def run(args):

    # env = Robot()
    env = ur5Env_1(is_render=True)

    algo = SACExpert(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device("cuda"),
        path=args.weight
    )

    buffer = collect_demo(
        env=env,
        algo=algo,
        buffer_size=args.buffer_size,
        device=torch.device("cuda"),
        std=args.std,
        p_rand=args.p_rand,
        seed=args.seed
    )

    buffer.save(os.path.join(
        'buffers',
        args.env_id,
        f'size{args.buffer_size}_std{args.std}_prand{args.p_rand}.pth'
    ))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--weight', type=str, default='D:\\pycharm_project\\xiangguangyu_SC_AIRL - Copy\\UR5-AIRL-master\\weights\\panda_stack_task_0.pth')
    # p.add_argument('--weight', type=str, default='D:\\pycharm_project\\xiangguangyu_SC_AIRL - Copy\\UR5-AIRL-master\\weights\\panda_stack_task_1.pth')
    p.add_argument('--env_id', type=str, default='panda_stack_task_1213_0')
    p.add_argument('--buffer_size', type=int, default=20000)
    p.add_argument('--std', type=float, default=0.01)
    p.add_argument('--p_rand', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    run(args)
