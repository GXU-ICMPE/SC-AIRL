import os
import argparse
from datetime import datetime
import torch
import pybullet_envs

from gail_airl_ppo.env import make_env
from gail_airl_ppo.buffer import SerializedBuffer, SerializedBuffer1
from gail_airl_ppo.algo import ALGOS
from gail_airl_ppo.trainer import Trainer
# from gym.envs.classic_control.ur5Env_1 import ur5Env_1
from env.panda_grasp_stack import ur5Env_1
import gym


def run(args):
    env = ur5Env_1()
    env_test = ur5Env_1(is_render=False)

    buffer_exp0 = SerializedBuffer(
        path=args.buffer0,
        device=torch.device('cuda')
    )

    buffer_exp1 = SerializedBuffer(
        path=args.buffer1,
        device=torch.device('cuda')
    )


    algo = ALGOS[args.algo](
        buffer_exp0=buffer_exp0,
        buffer_exp1=buffer_exp1,
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device('cuda'),
        seed=args.seed,
        path0=args.path0,
        pathc=args.pathc,
        rollout_length=args.rollout_length
    )

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, args.algo, f'seed{args.seed}-{time}')

    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        seed=args.seed
    )
    trainer.train()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--buffer0', type=str, required=True)
    p.add_argument('--buffer1', type=str, required=True)
    p.add_argument('--path0', type=str, required=True)
    p.add_argument('--pathc', type=str, required=True)
    p.add_argument('--rollout_length', type=int, default=10000)
    p.add_argument('--num_steps', type=int, default=10 ** 8)
    p.add_argument('--eval_interval', type=int, default=10000)
    p.add_argument('--env_id', type=str, default='panda_stack_sc-airl')
    p.add_argument('--algo', type=str, default='daairl')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    run(args)




# PPO 0927 stage=0