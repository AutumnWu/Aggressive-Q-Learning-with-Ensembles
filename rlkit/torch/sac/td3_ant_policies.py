import numpy as np
import torch
from torch import nn as nn

from rlkit.core.util import Wrapper
from rlkit.policies.base import ExplorationPolicy, Policy
from rlkit.torch.distributions import TanhNormal
from rlkit.torch.networks import Mlp
from rlkit.torch.core import np_ify
import rlkit.torch.pytorch_util as ptu


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class TanhGaussianPolicyTD3(Mlp, ExplorationPolicy):
    """
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```
    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """

    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            latent_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            action_limit=1.0,
            sigma=0.2,
            noise_clip=0.5,
            expl_noise=0.1,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )

        self.latent_dim = latent_dim
        self.action_limit = action_limit
        self.sigma = sigma
        self.noise_clip = noise_clip
        self.expl_noise = expl_noise

        last_hidden_size = obs_dim
        if len(hidden_sizes) > 0:
            last_hidden_size = hidden_sizes[-1]

    def get_action(self, obs, deterministic=False):
        actions = self.get_actions(obs, deterministic=deterministic)
        return actions[0, :], {}

    @torch.no_grad()
    def get_actions(self, obs, sigma=0.2, noise_clip=0.5, deterministic=False):
        outputs = self.forward(obs, deterministic=deterministic)[0]
        if not deterministic:
            #outputs = (
            #        outputs
            #        #+ torch.normal(0, torch.tensor(self.action_limit * self.expl_noise), size=outputs.size())
            #        + ptu.from_numpy(np.random.normal(0, self.action_limit * self.expl_noise, size= outputs.size()))
            #).clamp(-self.action_limit, self.action_limit)

            noise = (
                    torch.randn_like(outputs) * sigma
            ).clamp(-noise_clip, noise_clip)

            outputs = (
                    outputs + noise
            ).clamp(-self.action_limit, self.action_limit)

        return np_ify(outputs)

    def forward(
            self,
            obs,
            reparameterize=False,
            deterministic=False,
            return_log_prob=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)


        action = torch.tanh(mean)
        action = self.action_limit * action

        return (
            action, mean,
        )


class MakeDeterministic(Wrapper, Policy):
    def __init__(self, stochastic_policy):
        super().__init__(stochastic_policy)
        self.stochastic_policy = stochastic_policy

    def get_action(self, observation):
        return self.stochastic_policy.get_action(observation,
                                                 deterministic=True)

    def get_actions(self, observations):
        return self.stochastic_policy.get_actions(observations,
                                                  deterministic=True)
