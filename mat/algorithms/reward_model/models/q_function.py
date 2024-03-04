import numpy as np
import torch
import torch.nn as nn
from typing import Callable
from functools import partial
from torch.nn import functional as F


class FullyConnectedNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, inner_dim=256, orthogonal_init=False,
                 activations=nn.ReLU(), activation_final=None):
        super(FullyConnectedNetwork, self).__init__()
        ####################### set basic config
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.inner_dim = inner_dim
        self.orthogonal_init = orthogonal_init
        self.activations = activations
        self.activation_final = activation_final
        ####################### set model structure
        self.mlp = nn.Sequential(*[
            nn.Linear(self.input_dim, self.inner_dim),
            self.activations,
            nn.Linear(self.inner_dim, self.inner_dim),
            self.activations,
            nn.Linear(self.inner_dim, self.output_dim),
            self.activation_final,
        ])
        ####################### init model
        self.apply(self.init_MR_model)

    def init_MR_model(self, module):
        if self.orthogonal_init:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                nn.init.zeros_(module.bias)
        else:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        output = self.mlp(x)
        return output


class FullyConnectedQFunction(nn.Module):
    def __init__(self, observation_dim, action_dim, action_type, inner_dim=256, action_embd_dim=64,
                 orthogonal_init=False, activations='relu', activation_final='none', device='cpu'):
        super(FullyConnectedQFunction, self).__init__()
        ####################### set basic config
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.action_type = action_type
        self.inner_dim = inner_dim
        self.action_embd_dim = action_embd_dim
        self.orthogonal_init = orthogonal_init
        self.activations = activations
        self.activation_final = activation_final
        ####################### set model structure
        activations = {
            'none': nn.Identity(),
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
        }[self.activations]
        activation_final = {
            'none': nn.Identity(),
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
        }[self.activation_final]
        self.embed_action = nn.Linear(self.action_dim, self.action_embd_dim) if action_type == 'Discrete' else None
        self.input_dim = observation_dim + action_embd_dim if action_type == 'Discrete' else observation_dim + action_dim
        self.network = FullyConnectedNetwork(
            input_dim=self.input_dim, output_dim=1, inner_dim=self.inner_dim,
            orthogonal_init=self.orthogonal_init, activations=activations,
            activation_final=activation_final
        )
        ####################### copy model to correct device
        self.to(device)

    def forward(self, obs, actions):
        ####################### transform discrete action to onehot
        if self.action_type == 'Discrete':
            actions = F.one_hot(actions.squeeze(-1).long(), num_classes=self.action_dim).float()
            actions = self.embed_action(actions)
        x = torch.cat([obs, actions], dim=-1)
        x = self.network(x)
        return x.squeeze(-1)


