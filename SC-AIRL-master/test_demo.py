import os
import argparse
import torch
import gym
import numpy as np
from pybullet_envs.gym_locomotion_envs import AntBulletEnv, HopperBulletEnv

from gail_airl_ppo.env import make_env
from gail_airl_ppo.algo import SACExpert
from gail_airl_ppo.algo.ppo import PPOExpert1
from gail_airl_ppo.utils import display
from gym.envs.classic_control.ur5Env_1 import ur5Env_1
from env.panda_grasp_stack import ur5Env_1

# from realur5 import Robot


def run(args):

    # env = Robot()
    env = ur5Env_1(is_render=True)

    algo = PPOExpert1(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device("cuda"),
        path0=args.weight0,
        path1=args.weight1
    )

    buffer = display(
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
    p.add_argument('--weight0', type=str, default='UR5-AIRL-master/logs/panda_stack_sc-airl/actor.pth')
    p.add_argument('--weight1', type=str, default='UR5-AIRL-master/logs/panda_stack_sc-airl/actor1.pth')
    p.add_argument('--env_id', type=str, default='panda_stack_task_0')
    p.add_argument('--buffer_size', type=int, default=10000)
    p.add_argument('--std', type=float, default=0.05)
    p.add_argument('--p_rand', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    run(args)
