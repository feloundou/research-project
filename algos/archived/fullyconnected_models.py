import numpy as np
import torch
from torch import nn
from torch.nn import *  # export everthing
from torch.nn import functional as F
import numpy as np


class Flatten(nn.Module):
    """
    Flatten a tensor, e.g., between conv2d and linear layers.
    The maintainers FINALLY added this to torch.nn, but I am
    leaving it in for compatible for the moment.
    """

    def forward(self, x):
        return x.view(x.size()[0], -1)


class NoisyLinear(nn.Linear):
    """
    Implementation of Linear layer for NoisyNets
    https://arxiv.org/abs/1706.10295
    NoisyNets are a replacement for epsilon greedy exploration.
    Gaussian noise is added to the weights of the output layer, resulting in
    a stochastic policy. Exploration is implicitly learned at a per-state
    and per-action level, resulting in smarter exploration.
    """

    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        self.sigma_weight = nn.Parameter(
            torch.Tensor(out_features, in_features).fill_(sigma_init)
        )
        self.register_buffer(
            "epsilon_weight", torch.zeros(out_features, in_features))
        if bias:
            self.sigma_bias = nn.Parameter(
                torch.Tensor(out_features).fill_(sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        std = np.sqrt(3 / self.in_features)
        nn.init.uniform_(self.weight, -std, std)
        nn.init.uniform_(self.bias, -std, std)

    def perturb(self):
        torch.randn(self.epsilon_weight.size(), out=self.epsilon_weight)
        if self.bias is not None:
            torch.randn(self.epsilon_bias.size(), out=self.epsilon_bias)

    def forward(self, x):
        bias = self.bias

        if not self.training:
            return F.linear(x, self.weight, bias)

        if self.bias is not None:
            bias = bias + self.sigma_bias * self.epsilon_bias
        return F.linear(x, self.weight + self.sigma_weight * self.epsilon_weight, bias)


class NoisyFactorizedLinear(nn.Linear):
    """
    NoisyNet layer with factorized gaussian noise
    N.B. nn.Linear already initializes weight and bias to
    """

    def __init__(self, in_features, out_features, sigma_init=0.4, init_scale=3, bias=True):
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=bias)
        sigma_init = sigma_init / np.sqrt(in_features)
        self.sigma_weight = nn.Parameter(
            torch.Tensor(out_features, in_features).fill_(sigma_init)
        )
        self.register_buffer("epsilon_input", torch.zeros(1, in_features))
        self.register_buffer("epsilon_output", torch.zeros(out_features, 1))
        if bias:
            self.sigma_bias = nn.Parameter(
                torch.Tensor(out_features).fill_(sigma_init)
            )

    def perturb(self):
        torch.randn(self.epsilon_input.size(), out=self.epsilon_input)
        torch.randn(self.epsilon_output.size(), out=self.epsilon_output)

    def forward(self, input):
        if not self.training:
            return F.linear(input, self.weight, self.bias)

        def func(x): return torch.sign(x) * torch.sqrt(torch.abs(x))
        eps_in = func(self.epsilon_input)
        eps_out = func(self.epsilon_output)

        bias = self.bias
        if bias is not None:
            bias = bias + self.sigma_bias * eps_out.t()
        noise_v = torch.mul(eps_in, eps_out)
        return F.linear(input, self.weight + self.sigma_weight * noise_v, bias)


def perturb_noisy_layers(layer):
    if type(layer) == NoisyFactorizedLinear or type(layer) == NoisyLinear:
        layer.perturb()


class Linear0(nn.Linear):
    def reset_parameters(self):
        nn.init.constant_(self.weight, 0.0)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)


def kl_gaussian(mean1, log_var1, mean2, log_var2, epsilon=1e-4):
    # KL(p||q) when q == N(mean1, log_var1.exp())
    # and p == N(mean2, log_var2.exp())
    # epsilon is for numerical instability
    return 0.5 * (log_var2 - log_var1 +
                  log_var1.exp() / (log_var2.exp() + epsilon) +
                  (mean2 - mean1).pow(2) / (log_var2.exp() + epsilon)).sum(-1)


def kl_loss_vae(mean, log_var):
    # loss of KL(q||p) when q == N(mean, log_var.exp()), and p == N(0, 1)
    # See appendix B in https://arxiv.org/pdf/1312.6114.pdf
    return -0.5 * (1 + log_var - mean.pow(2) - log_var.exp()).mean()


def mmd_laplacian(samples1, samples2, sigma=10.0):
    """
    MMD constraint with Laplacian kernel for support matching in BEAR
    This code was stolen from: https://github.com/aviralkumar2907/BEAR/blob/master/algos.py
    sigma is set to 10.0 for hopper, cheetah and 20 for walker/ant
    """

    # Batch x num_samples x num_samples x dimension
    diff_x_x = samples1.unsqueeze(2) - samples1.unsqueeze(1)
    diff_x_x = torch.mean(
        (-(diff_x_x.abs()).sum(-1)/(2.0 * sigma)).exp(), dim=(1, 2))

    diff_x_y = samples1.unsqueeze(2) - samples2.unsqueeze(1)
    diff_x_y = torch.mean(
        (-(diff_x_y.abs()).sum(-1)/(2.0 * sigma)).exp(), dim=(1, 2))

    # Batch x num_samples x num_samples x dimension
    diff_y_y = samples2.unsqueeze(2) - samples2.unsqueeze(1)
    diff_y_y = torch.mean(
        (-(diff_y_y.abs()).sum(-1)/(2.0 * sigma)).exp(), dim=(1, 2))

    overall_loss = (diff_x_x + diff_y_y - 2.0 * diff_x_y + 1e-6).sqrt()
    return overall_loss


def mmd_gaussian(samples1, samples2, sigma=10.0):
    """
    MMD constraint with Gaussian Kernel support matching in BEAR
    This code was stolen from: https://github.com/aviralkumar2907/BEAR/blob/master/algos.py
    sigma is set to 10.0 for hopper, cheetah and 20 for walker/ant
    """

    # Batch x num_samples x num_samples x dimension
    diff_x_x = samples1.unsqueeze(
        2) - samples1.unsqueeze(1)
    diff_x_x = torch.mean(
        (-(diff_x_x.pow(2)).sum(-1)/(2.0 * sigma)).exp(), dim=(1, 2))

    diff_x_y = samples1.unsqueeze(2) - samples2.unsqueeze(1)
    diff_x_y = torch.mean(
        (-(diff_x_y.pow(2)).sum(-1)/(2.0 * sigma)).exp(), dim=(1, 2))

    # Batch x num_samples x num_samples x dimension
    diff_y_y = samples2.unsqueeze(2) - samples2.unsqueeze(1)
    diff_y_y = torch.mean(
        (-(diff_y_y.pow(2)).sum(-1)/(2.0 * sigma)).exp(), dim=(1, 2))

    overall_loss = (diff_x_x + diff_y_y - 2.0 * diff_x_y + 1e-6).sqrt()
    return overall_loss


def weighted_mse_loss(input, target, weight, reduction='mean'):
    loss = (weight * ((target - input) ** 2))
    return torch.mean(loss) if reduction == 'mean' else torch.sum(loss)


def fc_q(env, hidden1=400, hidden2=300):
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0] +
                  env.action_space.shape[0], hidden1),
        nn.LeakyReLU(),
        nn.Linear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.Linear(hidden2, 1),
    )


def fc_v(env, hidden1=400, hidden2=300):
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0], hidden1),
        nn.LeakyReLU(),
        nn.Linear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.Linear(hidden2, 1),
    )


def fc_deterministic_policy(env, hidden1=400, hidden2=300):
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0], hidden1),
        nn.LeakyReLU(),
        nn.Linear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.Linear(hidden2, env.action_space.shape[0]),
    )


def fc_deterministic_noisy_policy(env, hidden1=400, hidden2=300):
    return nn.Sequential(
        nn.NoisyFactorizedLinear(env.state_space.shape[0], hidden1),
        nn.LeakyReLU(),
        nn.NoisyFactorizedLinear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.NoisyFactorizedLinear(hidden2, env.action_space.shape[0]),
    )


def fc_soft_policy(env, hidden1=400, hidden2=300):
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0], hidden1),
        nn.LeakyReLU(),
        nn.Linear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.Linear(hidden2, env.action_space.shape[0] * 2),
    )


def fc_actor_critic(env, hidden1=400, hidden2=300):
    features = nn.Sequential(
        # nn.Linear(env.state_space.shape[0], hidden1),  # Tyna note
        nn.Linear(env.observation_space.shape[0], hidden1),
        nn.LeakyReLU(),
    )

    v = nn.Sequential(
        nn.Linear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.Linear(hidden2, 1)
    )

    policy = nn.Sequential(
        nn.Linear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.Linear(hidden2, env.action_space.shape[0] * 2)
    )

    return features, v, policy


def fc_discriminator(env, hidden1=400, hidden2=300):
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0] + env.action_space.shape[0],
                  hidden1),
        nn.LeakyReLU(),
        nn.Linear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.Linear(hidden2, 1),
        nn.Sigmoid())


def fc_bcq_encoder(env, latent_dim=32, hidden1=400, hidden2=300):
    # output mean and log_var
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0] +
                  env.action_space.shape[0], hidden1),
        nn.LeakyReLU(),
        nn.Linear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.Linear(hidden2, latent_dim * 2)
    )


def fc_bcq_decoder(env, latent_dim=32, hidden1=300, hidden2=400):
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0] + latent_dim, hidden1),
        nn.LeakyReLU(),
        nn.Linear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.Linear(hidden2, env.action_space.shape[0])
    )


def fc_bcq_deterministic_policy(env, hidden1=400, hidden2=300):
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0] +
                  env.action_space.shape[0], hidden1),
        nn.LeakyReLU(),
        nn.Linear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.Linear(hidden2, env.action_space.shape[0]),
    )


def fc_reward(env, hidden1=400, hidden2=300):
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0] +
                  env.action_space.shape[0], hidden1),
        nn.LeakyReLU(),
        nn.Linear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.Linear(hidden2, 1)
    )


def fc_dynamics(env, hidden1=500, hidden2=500):
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0] +
                  env.action_space.shape[0], hidden1),
        nn.LeakyReLU(),
        nn.Linear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.Linear(hidden2, env.state_space.shape[0]),
    )