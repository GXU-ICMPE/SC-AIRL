import os
import argparse
from datetime import datetime
import torch

from gail_airl_ppo.env import make_env
from gail_airl_ppo.algo import SAC, PPON
from gail_airl_ppo.trainer import Trainer
import pybullet_data
from env.panda_grasp_stack import ur5Env_1
import pybullet_envs
# from realur5 import Robot


def run(args):
    env = ur5Env_1(is_render=True)
    env_test = ur5Env_1(is_render=False)

    algo = SAC(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device("cuda"),
        seed=args.seed,
    )

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, 'sac', f'seed{args.seed}-{time}')

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
    p.add_argument('--num_steps', type=int, default=10000000)
    p.add_argument('--eval_interval', type=int, default=2048)
    p.add_argument('--env_id', type=str, default='panda_grasp_1210')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    run(args)
