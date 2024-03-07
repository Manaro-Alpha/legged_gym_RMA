import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCritic_RMA(nn.Module):
    def __init__(self,num_actor_obs, num_critic_obs, num_actions, encoder_in_dim, encoder_out_dim, state_hist_in_dim, num_hist, activation = "lrelu", enc_activation = "tanh",init_noise_std=1.0):
        super().__init__()
        
        self.activation = get_activation(activation)
        self.activation_encoder = get_activation(enc_activation)
        actor_in_dim = num_actor_obs + encoder_out_dim
        self.actor_out_dim =num_actions
        critic_in_dim = num_critic_obs
        self.tsteps = num_hist

        self.actor = nn.Sequential(
            nn.Linear(actor_in_dim,512),
            self.activation,
            nn.Linear(512,256),
            self.activation,
            nn.Linear(256,128),
            self.activation,
            nn.Linear(128,num_actions)
        )

        self.critic = nn.Sequential(
            nn.Linear(critic_in_dim,512),
            self.activation,
            nn.Linear(512,256),
            self.activation,
            nn.Linear(256,128),
            self.activation,
            nn.Linear(128,1)
        )

        self.encoder = nn.Sequential(
            nn.Linear(encoder_in_dim,256),
            self.activation,
            nn.Linear(256,128),
            self.activation,
            nn.Linear(128,encoder_out_dim)
        )
        
        self.history_encoder_mlp = nn.Sequential(
            nn.Linear(state_hist_in_dim,32),
            self.activation,
        )

        self.history_encoder_conv = nn.Sequential(
            nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 8, stride = 4),
            self.activation,
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1),
            self.activation,
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1),
            nn.Flatten()
        )

        self.history_encoder_out = nn.Sequential(
            nn.Linear(96,encoder_out_dim),
            self.activation
        )

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, encoder_observations, **kwargs):
        z = self.encoder(encoder_observations)
        observations = torch.cat((observations,z),dim=-1)
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def act_hist_encoder(self,observation,encoder_observation,observation_history,action_history):
        observation_history = torch.cat((observation_history,action_history),dim=-1)
        hist_shape = observation_history.shape
        print("###################################",hist_shape)
        z1 = self.history_encoder_mlp(observation_history)
        z2 = self.history_encoder_conv(z1.reshape(z1))
        z = self.history_encoder_out(z2)
        z_target = self.encoder(encoder_observation)
        obs = torch.cat((observation,z),dim=-1)
        self.update_distribution(obs)
        return self.distribution.sample(),z,z_target
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations,encoder_observations):
        z = self.encoder(encoder_observations)
        observations = torch.cat((observations,z),dim=-1)
        actions_mean = self.actor(observations)
        return actions_mean
    
    def act_inference_dagger(self,obs,obs_history,action_history):
        tsteps = self.tsteps
        obs_history = obs_history.reshape(obs.shape[0],tsteps,obs.shape[1])
        action_history = action_history.reshape(obs.shape[0],tsteps,self.actor_out_dim)
        
        observation_history = torch.cat((obs_history,action_history),dim=-1)
        hist_shape = observation_history.shape
        # print("###################################",hist_shape)
        z1 = self.history_encoder_mlp(observation_history)
        # print(z1.shape)
        z2 = self.history_encoder_conv(z1.reshape(hist_shape[0],32,30))
        z = self.history_encoder_out(z2)
        obs = torch.cat((obs,z),dim=-1)
        actions_mean = self.actor(obs)
        return actions_mean ## take mean or sample from a distribution??


    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
    


class AdaptationModule(nn.Module):
    def __init__(self,actor_critic:ActorCritic_RMA,num_actor_obs, num_critic_obs, num_actions, encoder_in_dim, encoder_out_dim, state_hist_in_dim, num_hist, activation = "elu", enc_activation = "tanh",init_noise_std=1.0):
        super().__init__()
        # self.actor_critic = ActorCritic_RMA(num_actor_obs, num_critic_obs, num_actions, encoder_in_dim, encoder_out_dim, state_hist_in_dim, num_hist, activation = "lrelu", enc_activation = "tanh",init_noise_std=1.0)
        self.actor_critic = actor_critic
        self.tsteps = num_hist    
        self.activation_encoder = get_activation(enc_activation)
        self.activation = get_activation(activation)
        self.actor_out_dim = num_actions
        
        self.history_encoder_mlp = nn.Sequential(
            nn.Linear(state_hist_in_dim,32),
            self.activation,
        )

        self.history_encoder_conv = nn.Sequential(
            nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 8, stride = 4),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1),
            nn.LeakyReLU(),
            nn.Flatten()
        )

        self.history_encoder_out = nn.Sequential(
            nn.Linear(96,encoder_out_dim),
        )

    def act_hist_encoder(self,observation,encoder_observation,observation_history,action_history):
        observation_history = torch.cat((observation_history,action_history),dim=-1)
        hist_shape = observation_history.shape
        # print("###################################",hist_shape)
        z1 = self.history_encoder_mlp(observation_history)
        z2 = self.history_encoder_conv(z1.reshape(z1))
        z = self.history_encoder_out(z2)
        z_target = self.encoder(encoder_observation)
        obs = torch.cat((observation,z),dim=-1)
        ActorCritic_RMA.update_distribution(obs)
        return ActorCritic_RMA.distribution.sample(),z,z_target
    
    def act_inference_dagger(self,obs,encoder_observation,obs_history,action_history):
        tsteps = self.tsteps
        # print('obs shape: ',obs.shape,"\nobs hist shape:",obs_history.shape, "action hist shape: ",action_history.shape)
        obs_history = obs_history.reshape(obs.shape[0],tsteps,obs.shape[1])
        action_history = action_history.reshape(obs.shape[0],tsteps,self.actor_out_dim)
        
        observation_history = torch.cat((obs_history,action_history),dim=-1)
        hist_shape = observation_history.shape
        # print("###################################",hist_shape)
        z1 = self.history_encoder_mlp(observation_history)
        # print(z1.shape)
        z2 = self.history_encoder_conv(z1.reshape(hist_shape[0],32,tsteps))
        z = self.history_encoder_out(z2)
        z_target = self.actor_critic.encoder(encoder_observation)
        obs = torch.cat((obs,z),dim=-1)
        actions_mean = self.actor_critic.actor(obs)
        return actions_mean,z,z_target ## take mean or sample from a distribution??  











def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
