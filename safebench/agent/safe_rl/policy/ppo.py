import numpy as np
import torch
import torch.nn as nn
from safebench.agent.safe_rl.policy.base_policy import Policy
from safebench.agent.safe_rl.policy.model.mlp_ac import MLPCategoricalActor, MLPGaussianActor, mlp
from safebench.util.torch_util import (count_vars, get_device_name, to_device, to_ndarray, to_tensor)
from torch.optim import Adam


class PPO(Policy):
    def __init__(self, config, logger):
        r'''
        Promximal Policy Optimization (PPO)

        Args:
        @param env : The environment must satisfy the OpenAI Gym API.
        @param logger: Log useful informations, and help to save model
        @param actor_lr, critic_lr (float): Learning rate for policy and Q-value learning.
        @param ac_model: the actor critic model name

        @param clip_ratio (float): Hyperparameter for clipping in the policy objective. Roughly: how far can the new policy go from the old policy while still profiting (improving the objective function)? The new policy can still go farther than the clip_ratio says, but it doesn't help on the objective anymore. (Usually small, 0.1 to 0.3.) Typically denoted by :math:`\epsilon`. 
        @param target_kl (float): Roughly what KL divergence we think is appropriate between new and old policies after an update. This will get used for early stopping. (Usually small, 0.01 or 0.05.)
        @param train_actor_iters, train_critic_iters (int): Training iterations for actor and critic
        '''
        super().__init__()

        self.logger = logger
        self.clip_ratio = config['clip_ratio']
        self.target_kl = config['target_kl']
        self.train_actor_iters = config['train_actor_iters']
        self.train_critic_iters = config['train_critic_iters']
        self.actor_lr = config['actor_lr']
        self.critic_lr = config['critic_lr']
        self.gamma = config['gamma']
        self.hidden_sizes = config['hidden_sizes']

        ################ create actor critic model ###############
        self.obs_dim = config['ego_state_dim']
        self.act_dim = config['ego_action_dim']
        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        self.act_lim = config['ego_action_limit']

        # if ac_model.lower() == "mlp":
        actor = MLPGaussianActor(self.obs_dim, self.act_dim, -self.act_lim, self.act_lim, self.hidden_sizes, nn.ReLU)
        critic = mlp([self.obs_dim] + list(self.hidden_sizes) + [1], nn.ReLU)
        # else:
        #     raise ValueError(f"{ac_model} ac model does not support.")

        # Set up optimizer and device
        self._ac_training_setup(actor, critic)

        # Set up model saving
        # self.save_model()

        # Count variables
        var_counts = tuple(
            count_vars(module) for module in [self.actor, self.critic])
        logger.log('\nNumber of parameters: \t actor: %d, \t critic: %d\n' % var_counts)

    def _ac_training_setup(self, actor, critic):
        self.actor, self.critic = to_device([actor, critic], get_device_name())

        # Set up optimizers for policy and value function
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr)

    def act(self, obs, deterministic=False, with_logprob=False):
        '''
        Given a single obs, return the action, value, logp.
        This API is used to interact with the env.

        @param obs, 1d ndarray
        @param eval, evaluation mode
        @return act, value, logp, 1d ndarray
        '''
        obs = to_tensor(obs).reshape(1, -1)
        with torch.no_grad():
            _, a, logp_a = self.actor_forward(obs, deterministic=deterministic)
            v = self.critic_forward(self.critic, obs)
        # squeeze them to the right shape
        a, v, logp_a = np.squeeze(to_ndarray(a), axis=0), np.squeeze(to_ndarray(v)), np.squeeze(to_ndarray(logp_a))
        return a, v, logp_a

    def learn_on_batch(self, data: dict):
        '''
        Given a batch of data, train the policy
        data keys: (obs, act, ret, adv, logp)
        '''
        self._update_actor(data)

        LossV, DeltaLossV = self._update_critic(self.critic, data["obs"], data["ret"], self.critic_optimizer)
        # Log critic update info
        self.logger.store(LossV=LossV, DeltaLossV=DeltaLossV)

    def critic_forward(self, critic, obs):
        # Critical to ensure value has the right shape.
        # Without this, the training stability will be greatly affected!
        # For instance, shape [3] - shape[3,1] = shape [3, 3] instead of shape [3]
        return torch.squeeze(critic(obs), -1)

    def actor_forward(self, obs, act=None, deterministic=False):
        r''' 
        Return action distribution and action log prob [optional].
        @param obs, [tensor], (batch, obs_dim)
        @param act, [tensor], (batch, act_dim). If None, log prob is None
        @return pi, [torch distribution], (batch,)
        @return a, [torch distribution], (batch, act_dim)
        @return logp, [tensor], (batch,)
        '''
        pi, a, logp = self.actor(obs, act, deterministic)
        return pi, a, logp

    def _update_actor(self, data):
        obs, act, adv, logp_old = to_tensor(data['obs']), to_tensor(data['act']), to_tensor(data['adv']), to_tensor(data['logp'])

        def policy_loss():
            pi, _, logp = self.actor_forward(obs, act)
            ratio = torch.exp(logp - logp_old)
            clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
            loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

            # Useful extra info
            approx_kl = (logp_old - logp).mean().item()

            ent = pi.entropy().mean().item()
            clipped = ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio)
            clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
            pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

            return loss_pi, pi_info

        pi_l_old, pi_info_old = policy_loss()

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_actor_iters):
            self.actor_optimizer.zero_grad()
            loss_pi, pi_info = policy_loss()
            if i == 0 and pi_info['kl'] >= 1e-7:
                print("**" * 20)
                print("1st kl: ", pi_info['kl'])
            if pi_info['kl'] > 1.5 * self.target_kl:
                self.logger.log(
                    'Early stopping at step %d due to reaching max kl.' % i)
                break
            loss_pi.backward()
            self.actor_optimizer.step()

        # Log actor update info
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']

        self.logger.store(
            StopIter=i,
            LossPi=to_ndarray(pi_l_old),
            KL=kl,
            Entropy=ent,
            ClipFrac=cf,
            DeltaLossPi=(to_ndarray(loss_pi) - to_ndarray(pi_l_old))
        )

    def _update_critic(self, critic, obs, ret, critic_optimizer):
        obs, ret = to_tensor(obs), to_tensor(ret)

        def critic_loss():
            ret_pred = self.critic_forward(critic, obs)
            return ((ret_pred - ret)**2).mean()

        loss_old = critic_loss().item()

        # Value function learning
        for i in range(self.train_critic_iters):
            critic_optimizer.zero_grad()
            loss_critic = critic_loss()
            loss_critic.backward()
            critic_optimizer.step()

        return loss_old, to_ndarray(loss_critic) - loss_old

    # def save_model(self):
    #     self.logger.setup_pytorch_saver((self.actor.state_dict(), self.critic.state_dict()))

    def load_model(self, path):
        actor_state_dict, critic_state_dict = torch.load(path)
        self.actor.load_state_dict(actor_state_dict)
        self.actor.eval()
        self.critic.load_state_dict(critic_state_dict)
        self.critic.eval()
        self._ac_training_setup(self.actor, self.critic)
        # self.save_model()
