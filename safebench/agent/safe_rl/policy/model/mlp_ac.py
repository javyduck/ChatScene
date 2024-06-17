from safebench.util.torch_util import to_device, to_tensor
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

# "normal" or "uniform" or None
INIT_METHOD = "normal"


def mlp(sizes, activation, output_activation=nn.Identity):
    if INIT_METHOD == "normal":
        initializer = nn.init.xavier_normal_
    elif INIT_METHOD == "uniform":
        initializer = nn.init.xavier_uniform_
    else:
        initializer = None
    bias_init = 0.0
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layer = nn.Linear(sizes[j], sizes[j + 1])
        if initializer is not None:
            # init layer weight
            initializer(layer.weight)
            nn.init.constant_(layer.bias, bias_init)
        layers += [layer, act()]
    return nn.Sequential(*layers)


class MLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit=1):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        return self.act_limit * self.pi(obs)


class MLPGaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, action_low, action_high, hidden_sizes,
                 activation):
        super().__init__()
        self.action_low = torch.nn.Parameter(to_tensor(action_low)[None, ...],
                                             requires_grad=False)  # (1, act_dim)
        self.action_high = torch.nn.Parameter(to_tensor(action_high)[None, ...],
                                              requires_grad=False)  # (1, act_dim)
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = torch.sigmoid(self.mu_net(obs))
        mu = self.action_low + (self.action_high - self.action_low) * mu
        std = torch.exp(self.log_std)
        return mu, Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(
            axis=-1)  # Last axis sum needed for Torch Normal distribution

    def forward(self, obs, act=None, deterministic=False):
        '''
        Produce action distributions for given observations, and
        optionally compute the log likelihood of given actions under
        those distributions.
        If act is None, sample an action
        '''
        mu, pi = self._distribution(obs)
        if act is None:
            act = pi.sample()
        if deterministic:
            act = mu
        logp_a = self._log_prob_from_distribution(pi, act)
        return pi, act, logp_a


class MLPCategoricalActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

    def forward(self, obs, act=None):
        '''
        Produce action distributions for given observations, and
        optionally compute the log likelihood of given actions under
        those distributions.
        If act is None, sample an action
        '''
        pi = self._distribution(obs)
        if act is None:
            act = pi.sample()
        # print(act.shape)
        # print(act[:10])
        logp_a = self._log_prob_from_distribution(pi, act)
        return pi, act, logp_a


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SquashedGaussianMLPActor(nn.Module):
    '''
    Probablistic actor, can also be used as a deterministic actor
    '''
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)

    def forward(self,
                obs,
                deterministic=False,
                with_logprob=True,
                with_distribution=False):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # print("actor: ", torch.sum(mu), torch.sum(std))

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(
                axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)

        if with_distribution:
            return pi_action, logp_pi, pi_distribution
        return pi_action, logp_pi


class CholeskyGaussianActor(nn.Module):
    """
    Policy network
    :param env: OpenAI gym environment
    """
    COV_MIN = 1e-4  # last exp is 1e-2
    MEAN_CLAMP_MIN = -5
    MEAN_CLAMP_MAX = 5
    COV_CLAMP_MIN = -5
    COV_CLAMP_MAX = 20

    def __init__(self, obs_dim, act_dim, action_low, action_high, hidden_sizes,
                 activation):
        super(CholeskyGaussianActor, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.action_low = torch.nn.Parameter(to_tensor(action_low)[None, ...],
                                             requires_grad=False)  # (1, act_dim)
        self.action_high = torch.nn.Parameter(to_tensor(action_high)[None, ...],
                                              requires_grad=False)  # (1, act_dim)

        if INIT_METHOD == "normal":
            initializer = nn.init.xavier_normal_
        elif INIT_METHOD == "uniform":
            initializer = nn.init.xavier_uniform_
        else:
            initializer = None

        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mean_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.cholesky_layer = nn.Linear(hidden_sizes[-1],
                                        (self.act_dim * (self.act_dim + 1)) // 2)
        if initializer is not None:
            # init layer weight
            initializer(self.mean_layer.weight)
            initializer(self.cholesky_layer.weight)
            nn.init.constant_(self.mean_layer.bias, 0.0)
            nn.init.constant_(self.cholesky_layer.bias, 0.0)

    def forward(self, state):
        """
        forwards input through the network
        :param state: (B, obs_dim)
        :return: mean vector (B, act_dim) and cholesky factorization of covariance matrix (B, act_dim, act_dim)
        """
        B = state.size(0)

        net_out = self.net(state)

        clamped_mean = torch.clamp(self.mean_layer(net_out), self.MEAN_CLAMP_MIN,
                                   self.MEAN_CLAMP_MAX)
        mean = torch.sigmoid(clamped_mean)  # (B, act_dim)

        mean = self.action_low + (self.action_high - self.action_low) * mean
        cholesky_vector = torch.clamp(
            self.cholesky_layer(net_out), self.COV_CLAMP_MIN,
            self.COV_CLAMP_MAX)  # (B, (act_dim*(act_dim+1))//2)
        cholesky_diag_index = torch.arange(self.act_dim, dtype=torch.long) + 1
        # cholesky_diag_index = (cholesky_diag_index * (cholesky_diag_index + 1)) // 2 - 1
        cholesky_diag_index = torch.div(
            cholesky_diag_index *
            (cholesky_diag_index + 1), 2, rounding_mode='floor') - 1
        # add a small value to prevent the diagonal from being 0.
        cholesky_vector[:, cholesky_diag_index] = F.softplus(
            cholesky_vector[:, cholesky_diag_index]) + self.COV_MIN
        tril_indices = torch.tril_indices(row=self.act_dim, col=self.act_dim, offset=0)
        cholesky = to_device(
            torch.zeros(size=(B, self.act_dim, self.act_dim), dtype=torch.float32))
        cholesky[:, tril_indices[0], tril_indices[1]] = cholesky_vector
        return mean, cholesky

    # def action(self, state):
    #     """
    #     :param state: (obs_dim,)
    #     :return: an action
    #     """
    #     with torch.no_grad():
    #         mean, cholesky = self.forward(state[None, ...])
    #         action_distribution = MultivariateNormal(mean, scale_tril=cholesky)
    #         action = action_distribution.sample()
    #     return action[0]


class EnsembleQCritic(nn.Module):
    '''
    An ensemble of Q network to address the overestimation issue.
    '''
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, num_q=2):
        super().__init__()
        assert num_q >= 1, "num_q param should be greater than 1"
        self.q_nets = nn.ModuleList([
            mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], nn.ReLU)
            for i in range(num_q)
        ])

    def forward(self, obs, act):
        # Squeeze is critical to ensure value has the right shape.
        # Without squeeze, the training stability will be greatly affected!
        # For instance, shape [3] - shape[3,1] = shape [3, 3] instead of shape [3]
        data = torch.cat([obs, act], dim=-1)
        return [torch.squeeze(q(data), -1) for q in self.q_nets]

    def predict(self, obs, act):
        q_list = self.forward(obs, act)
        qs = torch.vstack(q_list)  # [num_q, batch_size]
        return torch.min(qs, dim=0).values, q_list

    def loss(self, target, q_list=None):
        losses = [((q - target)**2).mean() for q in q_list]
        return sum(losses)
