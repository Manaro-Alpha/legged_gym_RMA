from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, export_adapt_mod_as_jit, task_registry, Logger

import numpy as np
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs,obs_history,action_history = env.get_observations()
    encoder_obs = env.get_privileged_observations()
    # load policy
    train_cfg.runner.resume = True
    train_cfg.runner.load_run_policy = "Mar01_17-51-18_"
    train_cfg.runner.checkpoint = '1350'
    ppo_runner, train_cfg = task_registry.make_alg_runner_dagger(env=env, name=args.task, args=args, train_cfg=train_cfg)
    # policy = ppo_runner.get_inference_policy(device=env.device)
    policy = ppo_runner.get_inference_policy_teacher(device=env.device)
    
    if EXPORT_POLICY:
        path_policy = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        path_adapt_mod = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'adapt_mod')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path_policy)
        export_adapt_mod_as_jit(ppo_runner.alg.adapt_mod, path_adapt_mod)
        print('Exported policy as jit script to: ', path_policy, " ", path_adapt_mod)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 1000 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    for i in range(10*int(env.max_episode_length)):
        actions,_,_ = policy(obs.detach(),encoder_obs.detach(),obs_history.detach(),action_history.detach())
        obs, encoder_obs, obs_history, action_history, rews, dones, infos = env.step(actions.detach())
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy(),
                    "actions0": actions[robot_index,0].cpu().detach().numpy(),
                    "actions1": actions[robot_index,1].cpu().detach().numpy(),
                    "actions2": actions[robot_index,2].cpu().detach().numpy(),
                    "actions3": actions[robot_index,3].cpu().detach().numpy(),
                    "actions4": actions[robot_index,4].cpu().detach().numpy(),
                    "actions5": actions[robot_index,5].cpu().detach().numpy(),
                    "actions6": actions[robot_index,6].cpu().detach().numpy(),
                    "actions7": actions[robot_index,7].cpu().detach().numpy(),
                    "actions8": actions[robot_index,8].cpu().detach().numpy(),
                    "actions9": actions[robot_index,9].cpu().detach().numpy(),
                    "actions10": actions[robot_index,10].cpu().detach().numpy(),
                    "actions11": actions[robot_index,11].cpu().detach().numpy()
                }
            )
        elif i==stop_state_log:
            logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)