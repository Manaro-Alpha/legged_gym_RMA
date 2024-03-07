import torch
import numpy as np

from rsl_rl.utils import split_and_pad_trajectories

class RolloutStorage_Dagger:
    class Transition:
        def __init__(self):
            self.observations = None
            self.encoder_observations = None
            self.observation_history = None
            self.action_history = None
            self.actions = None
            self.env_vector = None
            # self.rewards = None
            # self.dones = None

            # self.values = None
            # self.actions_log_prob = None
            # self.action_mean = None
            # self.action_sigma = None
            # self.hidden_states = None
        
        def clear(self):
            self.__init__()
    
    def __init__(self, num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, actions_shape, obs_history_shape, action_history_shape, env_vector_shape,device='cpu'):

        self.device = device

        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        
        self.actions_shape = actions_shape

        # Core
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        if privileged_obs_shape[0] is not None:
            self.privileged_observations = torch.zeros(num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device)
        else:
            self.privileged_observations = None
        
        self.observation_hist = torch.zeros(num_transitions_per_env, num_envs, *obs_history_shape, device=self.device)
        self.action_hist = torch.zeros(num_transitions_per_env, num_envs, *action_history_shape, device=self.device)
        # self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.env_vector = torch.zeros(num_transitions_per_env, num_envs, *env_vector_shape, device=self.device)
        # self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For PPO
        # self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        # self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        # self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        # self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        # self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        # self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # rnn
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        self.step = 0

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None: self.privileged_observations[self.step].copy_(transition.encoder_observations)
        self.observation_hist[self.step].copy_(transition.observation_history)
        self.action_hist[self.step].copy_(transition.action_history)
        self.actions[self.step].copy_(transition.actions)
        self.env_vector[self.step].copy_(transition.env_vector)
        # self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        # self.dones[self.step].copy_(transition.dones.view(-1, 1))
        # self.values[self.step].copy_(transition.values)
        # self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        # self.mu[self.step].copy_(transition.action_mean)
        # self.sigma[self.step].copy_(transition.action_sigma)
        # self._save_hidden_states(transition.hidden_states)
        self.step += 1

    def clear(self):
        self.step = 0

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
            batch_size = self.num_envs * self.num_transitions_per_env
            mini_batch_size = batch_size // num_mini_batches
            indices = torch.randperm(num_mini_batches*mini_batch_size, requires_grad=False, device=self.device)

            observations = self.observations.flatten(0, 1)
            if self.privileged_observations is not None:
                encoder_observations = self.privileged_observations.flatten(0, 1)
            else:
                encoder_observations = observations

            observation_history = self.observation_hist.flatten(0,1)
            action_history = self.action_hist.flatten(0,1)
            actions = self.actions.flatten(0, 1)
            env_vector = self.env_vector.flatten(0,1)
            # values = self.values.flatten(0, 1)
            # returns = self.returns.flatten(0, 1)
            # old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
            # advantages = self.advantages.flatten(0, 1)
            # old_mu = self.mu.flatten(0, 1)
            # old_sigma = self.sigma.flatten(0, 1)

            for epoch in range(num_epochs):
                for i in range(num_mini_batches):

                    start = i*mini_batch_size
                    end = (i+1)*mini_batch_size
                    batch_idx = indices[start:end]

                    obs_batch = observations[batch_idx]
                    encoder_observations_batch = encoder_observations[batch_idx]
                    observation_history_batch = observation_history[batch_idx]
                    action_history_batch = action_history[batch_idx]
                    actions_batch = actions[batch_idx]
                    env_vector_batch = env_vector[batch_idx]
                    # target_values_batch = values[batch_idx]
                    # returns_batch = returns[batch_idx]
                    # old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                    # advantages_batch = advantages[batch_idx]
                    # old_mu_batch = old_mu[batch_idx]
                    # old_sigma_batch = old_sigma[batch_idx]
                    yield obs_batch, encoder_observations_batch, observation_history_batch, action_history_batch, actions_batch, env_vector_batch


            