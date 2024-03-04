import torch
import functools
import numpy as np
import torch.nn as nn
import mat.algorithms.reward_model.models.ops as ops
from torch.nn import functional as F
from typing import Any


class SimpleLSTM(nn.Module):
    """A simple unidirectional LSTM."""
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return out

    # @functools.partial(
    #     nn.transforms.scan,
    #     variable_broadcast='params',
    #     in_axes=1, out_axes=1,
    #     split_rngs={'params': False})
    # @nn.compact
    # def __call__(self, carry, x):
    #     return nn.OptimizedLSTMCell()(carry, x)
    #
    # @staticmethod
    # def initialize_carry(batch_dims, hidden_size):
    #     # Use fixed random key since default state init fn is just zeros.
    #     return nn.OptimizedLSTMCell.initialize_carry(
    #         jax.random.PRNGKey(0), batch_dims, hidden_size)


class LSTMRewardModel(nn.Module):
    def __init__(self, config, observation_dim, action_dim, action_type, activation, activation_final, max_episode_steps, device):
        super(LSTMRewardModel, self).__init__()
        ####################### set basic config
        self.config_ = config
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.action_type = action_type
        self.activation = activation
        self.activation_final = activation_final
        self.max_episode_steps = max_episode_steps
        self.device = device
        self.config_.activation_function = self.activation
        self.config_.activation_final = self.activation_final
        self.vocab_size = self.config_.vocab_size
        self.max_pos = self.config_.n_positions
        self.embd_dim = self.config_.embd_dim
        self.action_embd_dim = self.config_.action_embd_dim
        self.embd_dropout = self.config_.embd_pdrop
        self.num_layers = self.config_.n_layer
        self.eps = self.config_.layer_norm_epsilon
        ####################### set model structure
        self.embed_action = nn.Linear(self.action_dim, self.action_embd_dim) if action_type == 'Discrete' else None
        self.input_dim = self.observation_dim + self.action_embd_dim if \
            action_type == 'Discrete' else self.observation_dim + self.action_dim
        self.head_mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.embd_dim),
            ops.apply_activation(self.activation),
            nn.Dropout(self.embd_dropout),
            nn.Linear(self.embd_dim, self.embd_dim // 2),
            ops.apply_activation(self.activation),
            nn.Dropout(self.embd_dropout),
            nn.Linear(self.embd_dim // 2, self.embd_dim // 2),
            ops.apply_activation(self.activation),
            nn.Dropout(self.embd_dropout),
        )
        self.lstm = SimpleLSTM(input_size=self.embd_dim // 2, hidden_size=self.embd_dim // 2, num_layers=1)
        self.tail_mlp = nn.Sequential(
            nn.Linear(self.embd_dim, self.embd_dim // 2),
            ops.apply_activation(self.activation),
            nn.Dropout(self.embd_dropout),
            nn.Linear(self.embd_dim // 2, self.embd_dim // 4),
            ops.apply_activation(self.activation),
            nn.Dropout(self.embd_dropout),
            nn.Linear(self.embd_dim // 4, self.embd_dim // 4),
            ops.apply_activation(self.activation),
            nn.Dropout(self.embd_dropout),
        )
        self.last = nn.Linear(self.embd_dim // 4, 1)
        ####################### copy mmodel to correct device
        self.to(device)

    def forward(self, states, actions):
        if self.action_type == 'Discrete':
            actions = F.one_hot(actions.squeeze(-1).long(), num_classes=self.action_dim).float()
            actions = self.embed_action(actions)
        x = torch.cat([states, actions], dim=-1)
        batch_size, seq_len, agent_num, embed_dim = \
            x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        # place agent dim to batch dim
        x = x.permute(0, 2, 1, 3).reshape(batch_size * agent_num, seq_len, embed_dim)
        x = self.head_mlp(x)
        lstm_out = self.lstm(x)
        x = torch.cat([x, lstm_out], dim=-1)
        x = self.tail_mlp(x)
        x = self.last(x)
        x = x.reshape(batch_size, agent_num, seq_len, 1).permute(0, 2, 1, 3)

        return x, lstm_out
