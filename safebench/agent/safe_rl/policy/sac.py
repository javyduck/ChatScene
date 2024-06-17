from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from safebench.agent.safe_rl.policy.base_policy import Policy
from safebench.agent.safe_rl.policy.model.mlp_ac import SquashedGaussianMLPActor, EnsembleQCritic
from safebench.util.torch_util import (count_vars, get_device_name, to_device, to_ndarray, to_tensor)
from torch.optim import Adam

class SAC(Policy):
    def __init__(self, config, logger):
        '''
        Args:
            @param env : The environment must satisfy the OpenAI Gym API.
            @param logger: Log useful informations, and help to save model
            @param actor_lr, critic_lr (float): Learning rate for policy and Q-value learning.
            @param ac_model: the actor critic model name
            @param alpha (float): Entropy regularization coefficient. (Equivalent to 
                inverse of reward scale in the original SAC paper.)
            @param gamma (float): Discount factor. (Always between 0 and 1.)
            @param polyak (float): Interpolation factor in polyak averaging for target 
            @param num_q (int): number of models in the q-ensemble critic.
        '''
        super().__init__()

        self.logger = logger
        self.alpha = config['alpha']
        self.gamma = config['gamma']
        self.polyak = config['polyak']
        self.actor_lr = config['actor_lr']
        self.critic_lr = config['critic_lr']
        self.hidden_sizes = config['hidden_sizes']

        ################ create actor critic model ###############
        self.obs_dim = config['ego_state_dim']
        self.act_dim = config['ego_action_dim']
        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        self.act_lim = config['ego_action_limit']

        if config['ac_model'].lower() == "mlp":
            actor = SquashedGaussianMLPActor(self.obs_dim, self.act_dim, self.hidden_sizes, nn.ReLU)
            critic = EnsembleQCritic(self.obs_dim, self.act_dim, self.hidden_sizes, nn.ReLU, num_q=config['num_q'])
        else:
            raise ValueError(f"{config['ac_model']} ac model does not support.")

        # Set up optimizer and target q models
        self._ac_training_setup(actor, critic)

        # Set up model saving
        # self.save_model()

        # Count variables
        var_counts = tuple(count_vars(module) for module in [self.actor, self.critic])
        self.logger.log('\nNumber of parameters: \t actor pi: %d, \t critic q: %d, \n' % var_counts)

    def _ac_training_setup(self, actor, critic):
        critic_targ = deepcopy(critic)
        self.actor, self.critic, self.critic_targ = to_device([actor, critic, critic_targ], get_device_name())
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.critic_targ.parameters():
            p.requires_grad = False

        # Set up optimizers for policy and value function
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr)

    def act(self, obs, deterministic=False, with_logprob=False):
        '''
            Given a single obs, return the action, logp.
            This API is used to interact with the env.

            @param obs (1d ndarray): observation
            @param deterministic (bool): True for evaluation mode, which returns the action with highest pdf (mean).
            @param with_logprob (bool): True to return log probability of the sampled action, False to return None
            @return act, logp, (1d ndarray)
        '''
        obs = to_tensor(obs).reshape(1, -1)
        with torch.no_grad():
            a, logp_a = self.actor_forward(obs, deterministic, with_logprob)
        # squeeze them to the right shape
        a, logp_a = np.squeeze(to_ndarray(a), axis=0), np.squeeze(to_ndarray(logp_a))
        return a, logp_a

    def learn_on_batch(self, data: dict):
        '''
            Given a batch of data, train the policy
            data keys: (obs, act, rew, obs_next, done)
        '''
        critic_loss = self._update_critic(data)
        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.critic.parameters():
            p.requires_grad = False

        actor_loss = self._update_actor(data)

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.critic.parameters():
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        self._polyak_update_target(self.critic, self.critic_targ)
        return critic_loss, actor_loss
    
    def critic_forward(self, critic, obs, act):
        # return the minimum q values and the list of all q_values
        return critic.predict(obs, act)

    def actor_forward(self, obs, deterministic=False, with_logprob=True):
        ''' 
            Return action distribution and action log prob [optional].
            @param obs, (tensor), [batch, obs_dim]
            @return a, (tensor), [batch, act_dim]
            @return logp, (tensor or None), (batch,)
        '''
        a, logp = self.actor(obs, deterministic, with_logprob)
        return a * self.act_lim, logp


    def _update_actor(self, data):
        '''
        Update the actor network
        '''
        def policy_loss():
            obs = to_tensor(data['state'])
            act, logp_pi = self.actor_forward(obs, False, True)
            q_pi, q_list = self.critic_forward(self.critic, obs, act)

            # Entropy-regularized policy loss
            loss_pi = (self.alpha * logp_pi - q_pi).mean()
            # Useful info for logging
            pi_info = dict(LogPi=to_ndarray(logp_pi))

            return loss_pi, pi_info

        self.actor_optimizer.zero_grad()
        loss_pi, pi_info = policy_loss()
        loss_pi.backward()
#         nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        return loss_pi.item()
    
    def _update_critic(self, data):
        def critic_loss():
            obs, act, reward, obs_next, done = to_tensor(
                data['state']), to_tensor(data['action']), to_tensor(
                    data['reward']), to_tensor(data['n_state']), to_tensor(data['done'])

            _, q_list = self.critic_forward(self.critic, obs, act)
            # Bellman backup for Q functions
            with torch.no_grad():
                # Target actions come from *current* policy
                act_next, logp_a_next = self.actor_forward(obs_next, deterministic=False, with_logprob=True)
                # Target Q-values
                q_pi_targ, _ = self.critic_forward(self.critic_targ, obs_next, act_next)
                backup = reward + self.gamma * (1 - done) * (q_pi_targ - self.alpha * logp_a_next)
            # MSE loss against Bellman backup
            loss_q = self.critic.loss(backup, q_list)
            # Useful info for logging
            q_info = dict()
            for i, q in enumerate(q_list):
                q_info["QVals" + str(i)] = to_ndarray(q)
            return loss_q, q_info

        # First run one gradient descent step for Q1 and Q2
        self.critic_optimizer.zero_grad()
        loss_critic, loss_q_info = critic_loss()
        loss_critic.backward()
#         nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        return loss_critic.item()
    
        # Log critic update info
        # Record things

    def _polyak_update_target(self, critic, critic_targ):
        with torch.no_grad():
            for p, p_targ in zip(critic.parameters(), critic_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def save_model(self, path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict()
        }, path)

    def load_model(self, path):
        if '.pt' in path:
            actor_state_dict, critic_state_dict = torch.load(path)
            self.actor.load_state_dict(actor_state_dict)
            self.critic.load_state_dict(critic_state_dict)
            self._ac_training_setup(self.actor, self.critic)
        else:
            checkpoint = torch.load(path)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])

            self._ac_training_setup(self.actor, self.critic)

            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

