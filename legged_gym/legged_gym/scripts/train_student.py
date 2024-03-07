from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch

def train(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs,obs_hist,action_hist = env.get_observations()
    train_cfg.runner.resume = True
    train_cfg.runner.teacher = False
    ppo_runner, train_cfg = task_registry.make_alg_runner_dagger(env=env, name=args.task, args=args, train_cfg=train_cfg)
    # policy = ppo_runner.get_inference_policy_teacher(device=env.device)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations,Policy=None, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args)

