import glob
import pickle as pkl
import lcm
import sys

from go1_gym_deploy.utils.deployment_runner import DeploymentRunner
from go1_gym_deploy.envs.lcm_agent import LCMAgent
from go1_gym_deploy.utils.cheetah_state_estimator import StateEstimator
from go1_gym_deploy.utils.command_profile import *
import numpy as np

import pathlib

lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")

def load_and_run_policy(label, experiment_name, max_vel=1.0, max_yaw_vel=1.0):
    # load agent
    dirs = glob.glob(f"../../runs/{label}/*")
    logdir = sorted(dirs)[0]

    with open(logdir+"/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        print(pkl_cfg.keys())
        cfg = pkl_cfg["Cfg"]
        print(cfg.keys())


    se = StateEstimator(lc)

    control_dt = 0.02
    command_profile = RCControllerProfile(dt=control_dt, state_estimator=se, x_scale=max_vel, y_scale=0.6, yaw_scale=max_yaw_vel)

    hardware_agent = LCMAgent(cfg, se, command_profile)
    se.spin()

    from go1_gym_deploy.envs.history_wrapper import HistoryWrapper
    hardware_agent = HistoryWrapper(hardware_agent)

    policy = load_policy(logdir)

    # load runner
    root = f"{pathlib.Path(__file__).parent.resolve()}/../../logs/"
    pathlib.Path(root).mkdir(parents=True, exist_ok=True)
    deployment_runner = DeploymentRunner(experiment_name=experiment_name, se=None,
                                         log_root=f"{root}/{experiment_name}")
    deployment_runner.add_control_agent(hardware_agent, "hardware_closed_loop")
    deployment_runner.add_policy(policy)
    deployment_runner.add_command_profile(command_profile)

    if len(sys.argv) >= 2:
        max_steps = int(sys.argv[1])
    else:
        max_steps = 10000000
    print(f'max steps {max_steps}')

    deployment_runner.run(max_steps=max_steps, logging=True)

def load_policy(logdir):
    actor = torch.jit.load(logdir + '/actor.pt')
    history_encoder_mlp = torch.jit.load(logdir + '/history_encoder_mlp.pt')
    history_encoder_conv = torch.jit.load(logdir + '/history_encoder_conv.pt')
    history_encoder_out = torch.jit.load(logdir + '/history_encoder_out.pt')

    def policy(obs, info):
        i = 0
        # latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        # action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        # info['latent'] = latent

        obs_1 = obs["obs"]
        obs_history = obs["obs_history"]
        action_history = obs["action_history"]
        obs_history.reshape(obs_1.shape[0],50,obs_1.shape[1])
        action_history.reshape(obs_1.shape[0],50,12)
        observation_history = torch.cat((obs_history,action_history),dim=-1)
        hist_shape = observation_history.shape
        z1 = history_encoder_mlp(observation_history)
        z2 = history_encoder_conv(z1.reshape(hist_shape[0],32,50))
        z = history_encoder_out(z2)
        action = actor(torch.cat(obs_1,z),dim=-1)
        return action

    return policy


if __name__ == '__main__':
    label = "gait-conditioned-agility/pretrain-v0/train"

    experiment_name = "example_experiment"

    load_and_run_policy(label, experiment_name=experiment_name, max_vel=3.5, max_yaw_vel=5.0)
