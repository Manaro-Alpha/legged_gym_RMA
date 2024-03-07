import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic,ActorCritic_RMA,AdaptationModule
from rsl_rl.storage import RolloutStorage_Dagger

class Dagger:
    actor_critic:ActorCritic_RMA
    adapt_mod:AdaptationModule
    def __init__(self,
                 actor_critic,
                 adapt_mod,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 learning_rate=1e-3,
                 device="cpu"
                 ):
        
        self.device = device
        self.learning_rate = learning_rate
        self.actor_critic = actor_critic
        self.adapt_mod = adapt_mod
        self.actor_critic.to(self.device)
        self.adapt_mod.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.adapt_mod.parameters(), lr=learning_rate)
        self.transition = RolloutStorage_Dagger.Transition()
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.max_grad_norm = 1.0
    
    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, obs_hist_shape,action_hist_shape,env_vector_shape):
        self.storage = RolloutStorage_Dagger(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, obs_hist_shape, action_hist_shape, env_vector_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
        self.adapt_mod.train()

    
    def train_mode(self):
        self.actor_critic.train()
        self.adapt_mod.train()

    def act(self, obs, encoder_obs, obs_history, action_history):
        # if self.actor_critic.is_recurrent:
        #     self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions,_,env_vector = self.adapt_mod.act_inference_dagger(obs,encoder_obs,obs_history,action_history)
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.encoder_observations = encoder_obs
        self.transition.observation_history = obs_history
        self.transition.action_history = action_history
        self.transition.env_vector = env_vector
        return self.transition.actions, self.transition.env_vector
    
    def process_env_step(self, dones):
        self.transition.dones = dones
        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    
    def update(self):
        mean_loss = 0
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, encoder_observations_batch, observation_history_batch, action_history_batch, actions_batch, env_vector_batch in generator:
            _,z,z_target = self.adapt_mod.act_inference_dagger(obs_batch,encoder_observations_batch,observation_history_batch,action_history_batch)
            z_target = env_vector_batch
            # z_target.detach()
            # print(z.shape,z_target.shape)
            # loss = nn.MSELoss()(z,z_target)
            loss = (env_vector_batch.detach() - z).norm(p=2, dim=1).mean()
            nn.utils.clip_grad_norm_(self.adapt_mod.parameters(), self.max_grad_norm)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            mean_loss += loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_loss /= num_updates
        self.storage.clear() #check whether correct or not

        return mean_loss
        