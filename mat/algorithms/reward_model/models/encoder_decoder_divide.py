import math
import torch
import numpy as np
import torch.nn as nn
import mat.algorithms.reward_model.models.ops as ops
from torch.nn import functional as F


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module


def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


class SelfAttention(nn.Module):
    def __init__(self, config, n_embd, n_head, device, masked=True):
        super(SelfAttention, self).__init__()
        assert n_embd % n_head == 0
        self.config = config
        self.n_embd = n_embd
        self.n_head = n_head
        self.device = device
        self.masked = masked
        # key, query, value projections for all heads
        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        self.value = init_(nn.Linear(n_embd, n_embd))
        # output projection
        self.proj = init_(nn.Linear(n_embd, n_embd))
        ######## model dropout
        self.atten_dropout = nn.Dropout(self.config.atten_dropout) if self.config.use_dropout else None
        self.resid_dropout = nn.Dropout(self.config.resid_dropout) if self.config.use_dropout else None

    def forward(self, x, m):
        B, L, D = x.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        q = self.query(x).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        v = self.value(x).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        # causal attention: (B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if self.masked:
            casual_mask = torch.tril(torch.ones((1, 1, L, L), device=self.device))
            casual_mask = casual_mask.to(dtype=torch.bool)
            attn_mask = ops.get_attention_mask(m, B)
            # att = att.masked_fill(casual_mask == 0, float('-inf'))
            att = torch.where(casual_mask, att, -10000.0)
            att = att + attn_mask
        att = F.softmax(att, dim=-1)
        if self.config.use_dropout:
            att = self.atten_dropout(att)
        y = att @ v  # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs)
        y = y.transpose(1, 2).contiguous().view(B, L, D)  # re-assemble all head outputs side by side
        # output projection
        y = self.proj(y)
        # use residual dropout if necessary
        if self.config.use_dropout:
            y = self.resid_dropout(y)
        return y


class EncodeBlock(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, config, n_embd, n_head, device='cpu', masked=True):
        super(EncodeBlock, self).__init__()
        self.masked = masked
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(config, n_embd, n_head, device, masked=masked)
        self.ffn = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )

    def forward(self, input):
        x, m = input[0], input[1]
        x = self.ln1(x + self.attn(x, m))
        x = self.ln2(x + self.ffn(x))

        return (x, m)


class DecodeBlock(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, config, n_embd, n_head, device='cpu'):
        super(DecodeBlock, self).__init__()
        ####################### set basic config
        self.config = config
        ####################### set model structure
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(config, n_embd, n_head, device, masked=False)
        self.ffn = nn.Sequential(
            init_(nn.Linear(n_embd, n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(n_embd, n_embd))
        )

    def forward(self, x, rep_enc):
        x = torch.cat([rep_enc, x], dim=-1) if not self.config.add_obs_action else rep_enc + x
        x = self.ln1(x + self.attn(x, None))
        x = self.ln2(x + self.ffn(x))

        return x


################################ model for agent layer
class AgentLayerEncoder(nn.Module):
    def __init__(self, config, obs_dim, n_embd, n_layer, n_head, device='cpu'):
        super(AgentLayerEncoder, self).__init__()
        ####################### set model structure
        self.obs_encoder = nn.Sequential(
            nn.LayerNorm(obs_dim), init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU())
        self.ln = nn.LayerNorm(n_embd)
        self.blocks = nn.Sequential(*[EncodeBlock(
            config, n_embd, n_head, device, masked=False) for _ in range(n_layer)])

    def forward(self, obs):
        # obs: (batch, n_agent, obs_dim)
        x = self.ln(self.obs_encoder(obs))
        rep = self.blocks((x, None))[0]
        return rep


class AgentLayerDecoder(nn.Module):
    def __init__(self, config, action_dim, n_embd, n_layer, n_head, action_type='Discrete', device='cpu'):
        super(AgentLayerDecoder, self).__init__()
        ####################### set basic config
        self.action_dim = action_dim
        self.action_type = action_type
        ####################### set model structure
        if action_type == 'Discrete':
            self.action_encoder = nn.Sequential(init_(nn.Linear(action_dim, n_embd, bias=False), activate=True), nn.GELU())
        else:
            self.action_encoder = nn.Sequential(init_(nn.Linear(action_dim, n_embd), activate=True), nn.GELU())
        self.ln = nn.LayerNorm(n_embd)
        self.blocks = nn.Sequential(*[DecodeBlock(
            config, n_embd if config.add_obs_action else 2 * n_embd, n_head, device) for _ in range(n_layer)])
        self.head = nn.Sequential(init_(nn.Linear(n_embd if config.add_obs_action else 2 * n_embd, n_embd), activate=True))

    def forward(self, actions, obs_rep):
        # action: (batch, n_agent, action_dim)
        # obs_rep: (batch, n_agent, n_embd)
        if self.action_type == 'Discrete':
            actions = F.one_hot(actions.squeeze(-1).long(), num_classes=self.action_dim).float()
        action_embeddings = self.action_encoder(actions)
        x = self.ln(action_embeddings)
        for block in self.blocks:
            x = block(x, obs_rep)
        logits = self.head(x)
        return logits


class AgentLayer(nn.Module):
    def __init__(self, config, obs_dim, action_dim, n_embd, n_layer, n_head, action_type, device='cpu'):
        super().__init__()
        ####################### set basic config
        self.config = config
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_embd = n_embd
        self.action_type = action_type
        ####################### set model structure
        if self.config.agent_layer_mlp:
            self.embed_action = nn.Linear(self.action_dim, self.n_embd) \
                if action_type == 'Discrete' else None
            self.input_dim = obs_dim + n_embd \
                if action_type == 'Discrete' else obs_dim + action_dim
            self.mlp = nn.Sequential(*[
                nn.Linear(self.input_dim, self.n_embd),
                nn.ReLU(),
                nn.Linear(self.n_embd, self.n_embd),
                nn.ReLU(),
            ])
        else:
            self.encoder = AgentLayerEncoder(
                config=config, obs_dim=obs_dim, n_embd=n_embd, n_layer=n_layer, n_head=n_head, device=device
            )
            self.decoder = AgentLayerDecoder(
                config=config, action_dim=action_dim, n_embd=n_embd, n_layer=n_layer, n_head=n_head,
                action_type=action_type, device=device,
            )

    def forward(self, obs, actions):
        # obs: (batch, n_agent, obs_dim)
        # action: (batch, n_agent, act_dim)
        if self.config.agent_layer_mlp:
            if self.action_type == 'Discrete':
                actions = F.one_hot(actions.squeeze(-1).long(), num_classes=self.action_dim).float()
                actions = self.embed_action(actions)
            x = torch.cat([obs, actions], dim=-1)
            x = self.mlp(x)
            return x
        else:
            obs_rep = self.encoder(obs)
            disc_value = self.decoder(actions, obs_rep)
            return disc_value


################################ model for time layer
class Encoder(nn.Module):
    def __init__(self, config, n_layer, n_embd, n_head, device='cpu'):
        super(Encoder, self).__init__()
        ####################### set model structure
        self.blocks = nn.Sequential(*[
            EncodeBlock(config, n_embd, n_head, device, masked=True) for _ in range(n_layer)
        ])
        # self.last = nn.Linear(n_embd, 1)

    def forward(self, input_embds, atten_mask):
        # x: (batch, seq_len, embd_dim)
        x = self.blocks((input_embds, atten_mask))
        # x = self.last(x[0])
        x = x[0]
        return x


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


class LSTMEncoder(nn.Module):
    def __init__(self, config, embd_dim, device='cpu'):
        super(LSTMEncoder, self).__init__()
        ####################### set basic config
        self.config = config
        self.embd_dim = embd_dim
        self.device = device
        self.embd_dropout = self.config.embd_pdrop
        ####################### set model structure
        self.head_mlp = nn.Sequential(
            nn.Linear(self.embd_dim, self.embd_dim),
            nn.ReLU(),
            nn.Dropout(self.embd_dropout),
            nn.Linear(self.embd_dim, self.embd_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.embd_dropout),
            nn.Linear(self.embd_dim // 2, self.embd_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.embd_dropout),
        )
        self.lstm = SimpleLSTM(input_size=self.embd_dim // 2, hidden_size=self.embd_dim // 2, num_layers=1)
        self.tail_mlp = nn.Sequential(
            nn.Linear(self.embd_dim, self.embd_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.embd_dropout),
            nn.Linear(self.embd_dim // 2, self.embd_dim // 4),
            nn.ReLU(),
            nn.Dropout(self.embd_dropout),
            nn.Linear(self.embd_dim // 4, self.embd_dim // 4),
            nn.ReLU(),
            nn.Dropout(self.embd_dropout),
        )
        # self.last = nn.Linear(self.embd_dim // 4, 1)

    def forward(self, x):
        # x: (batch, seq_len, embd_dim)
        x = self.head_mlp(x)
        lstm_out = self.lstm(x)
        x = torch.cat([x, lstm_out], dim=-1)
        x = self.tail_mlp(x)
        # x = self.last(x)
        return x


class TimeLayer(nn.Module):
    def __init__(self, config, n_layer, n_embd, n_agent, n_head, max_episode_steps, device='cpu'):
        super().__init__()
        ####################### set basic config
        self.config = config
        self.n_agent = n_agent
        ####################### set model structure
        if self.config.time_layer_mlp:
            self.encoder = nn.Linear(n_embd, 1)
        else:
            self.embed_timestep = nn.Embedding(max_episode_steps + 1, n_embd) \
                if not self.config.use_lstm else None
            if self.config.use_lstm:
                self.encoder = LSTMEncoder(config=config, embd_dim=n_embd, device=device)
            else:
                self.encoder = Encoder(
                    config=config, n_layer=n_layer, n_embd=n_embd, n_head=n_head, device=device
                )

    def forward(self, x, m, t):
        # x: (batch, seq_len, embd_dim)
        # logits: (batch, seq_len, 1)
        ####################### time layer use mlp
        if self.config.time_layer_mlp:
            return self.encoder(x)
        ####################### time layer use encoder or lstm
        if self.config.use_lstm:
            logits = self.encoder(x)
        else:
            batch_size, seq_length = t.shape[0], t.shape[1]
            embed_timestep = self.embed_timestep(t).unsqueeze(2).repeat(1, 1, self.n_agent, 1)
            embed_timestep = embed_timestep.permute(0, 2, 1, 3).reshape(batch_size * self.n_agent, seq_length, -1)
            x = x + embed_timestep
            logits = self.encoder(x, m)
        return logits


class MultiTransRewardDivideModel(nn.Module):
    def __init__(self, config, observation_dim, action_dim, n_agent, action_type, max_episode_steps, device='cpu'):
        super().__init__()
        ####################### set basic config
        self.config = config
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.embd_dim = config.embd_dim
        self.pref_attn_embd_dim = config.pref_attn_embd_dim
        self.n_agent = n_agent
        self.action_type = action_type
        self.max_episode_steps = max_episode_steps
        self.eps = config.layer_norm_epsilon
        self.device = device
        ####################### set agent layer model structure
        self.agent_layer = AgentLayer(
            config=config, obs_dim=observation_dim, action_dim=action_dim, n_embd=self.embd_dim,
            n_layer=config.n_layer, n_head=config.n_head, action_type=action_type, device=device,
        )
        ####################### set time layer model structure
        self.time_layer = TimeLayer(
            config=config, n_layer=config.n_layer, n_embd=self.embd_dim, n_agent=self.n_agent,
            n_head=config.n_head, max_episode_steps=max_episode_steps, device=device,
        )
        ############## set weight sum layer for time / agent layer
        if self.config.use_weighted_sum:
            self.agent_weight_head = nn.Linear(self.embd_dim // 4 if self.config.use_lstm else self.embd_dim, 3 * self.pref_attn_embd_dim)
            self.time_weight_head = nn.Linear(self.embd_dim // 4 if self.config.use_lstm else self.embd_dim, 2 * self.pref_attn_embd_dim + 1)
            self.attn_dropout = nn.Dropout(0.0)
        else:
            self.last = self.last = nn.Linear(
                self.embd_dim // 4 if self.config.use_lstm else self.embd_dim, 1)
        ####################### copy mmodel to correct device
        self.to(device)

    def forward(self, states, actions, timesteps, attn_mask=None):
        batch_size, seq_length, agent_num = states.shape[0], states.shape[1], states.shape[2]
        if attn_mask is None:
            attn_mask = torch.ones(
                (batch_size, seq_length, agent_num), dtype=torch.float32, device=self.device)
        # states: (batch_size, seq_length, agent_num, obs_dim) --> (batch_size * seq_length, agent_num, obs_dim)
        # actions (batch_size, seq_length, agent_num, act_dim) --> (batch_size * seq_length, agent_num, act_dim)
        ####################### agent layer
        states = states.reshape(batch_size * seq_length, agent_num, -1)
        actions = actions.reshape(batch_size * seq_length, agent_num, -1)
        agent_output = self.agent_layer(states, actions)
        # agent_output: (batch_size * seq_length, agent_num, embd_dim)
        ####################### time layer
        time_input = agent_output.reshape(batch_size, seq_length, agent_num, -1).permute(0, 2, 1, 3)
        # time_input: (batch_size * agent_num, seq_length, embd_dim)
        # attn_mask: (batch_size * agent_num, seq_length)
        time_input = time_input.reshape(batch_size * agent_num, seq_length, -1)
        attn_mask = attn_mask.permute(0, 2, 1).reshape(batch_size * agent_num, seq_length)
        # attn_mask: (batch_size * agent_num, seq_length, 1)
        time_output = self.time_layer(time_input, attn_mask, timesteps)
        # time_output: (batch_size * agent_num, seq_length, embd_dim) --> (batch_size, seq_length, agent_num, embd_dim)
        time_output = time_output.reshape(batch_size, agent_num, seq_length, -1).permute(0, 2, 1, 3)

        ############## weight sum layer
        if self.config.use_weighted_sum:
            ############## weight sum of agent layer
            # (batch_size * seq_length, agent_num, 2 * embd_dim + 1)
            x = self.agent_weight_head(time_output).reshape(batch_size * seq_length, agent_num, -1)
            # only one head, because value has 1 dim for predicting rewards directly.
            num_heads = 1
            # query: [B * seq_len, agent_num, embd_dim]
            # key: [B * seq_len, agent_num, embd_dim]
            # value: [B * seq_len, agent_num, embd_dim]
            qkv = torch.split(x, [self.pref_attn_embd_dim, self.pref_attn_embd_dim, self.pref_attn_embd_dim], dim=2)
            query, key, value = qkv[0], qkv[1], qkv[2]
            query = ops.split_heads(query, num_heads, self.pref_attn_embd_dim)
            key = ops.split_heads(key, num_heads, self.pref_attn_embd_dim)
            value = ops.split_heads(value, num_heads, self.pref_attn_embd_dim)
            # query: [B * seq_len, 1, agent_num, embd_dim]
            # key: [B * seq_len, 1, agent_num, embd_dim]
            # value: [B * seq_len, 1, agent_num, embd_dim]
            query_len, key_len = query.shape[-2], key.shape[-2]
            casual_mask = torch.ones((1, 1, agent_num, agent_num), device=self.device)[:, :, key_len - query_len:key_len, :key_len]
            casual_mask = casual_mask.to(dtype=torch.bool)
            out, _ = ops.attention(query, key, value, casual_mask, -1e-4, self.attn_dropout,
                                   scale_attn_weights=True, training=False, attn_mask=None, head_mask=None)
            # out: [B * seq_len, 1, agent_num, embd_dim] -> [B * seq_len, agent_num, embd_dim]
            out = ops.merge_heads(out, num_heads, self.pref_attn_embd_dim)
            ############## weight sum of time layer
            # out: [B * seq_len, agent_num, embd_dim] -> [B, agent_num, seq_len, embd_dim]
            out = out.reshape(batch_size, seq_length, agent_num, self.pref_attn_embd_dim).permute(0, 2, 1, 3)
            # out: [B, agent_num, seq_len, embd_dim] -> [B * agent_num, seq_len, embd_dim]
            out = out.reshape(batch_size * agent_num, seq_length, self.pref_attn_embd_dim)
            # only one head, because value has 1 dim for predicting rewards directly.
            num_heads = 1
            # query: [B * agent_num, seq_len, embd_dim]
            # key: [B * agent_num, seq_len, embd_dim]
            # value: [B * agent_num, seq_len, 1]
            x = self.time_weight_head(out)
            qkv = torch.split(x, [self.pref_attn_embd_dim, self.pref_attn_embd_dim, 1], dim=2)
            query, key, value = qkv[0], qkv[1], qkv[2]
            query = ops.split_heads(query, num_heads, self.pref_attn_embd_dim)
            key = ops.split_heads(key, num_heads, self.pref_attn_embd_dim)
            value = ops.split_heads(value, num_heads, 1)
            # query: [B * agent_num, 1, seq_len, embd_dim]
            # key: [B * agent_num, 1, seq_len, embd_dim]
            # value: [B * agent_num, 1, seq_len, 1]
            query_len, key_len = query.shape[-2], key.shape[-2]
            casual_mask = torch.ones((1, 1, seq_length, seq_length), device=self.device)[:, :, key_len - query_len:key_len, :key_len]
            casual_mask = casual_mask.to(dtype=torch.bool)
            out, _ = ops.attention(query, key, value, casual_mask, -1e-4, self.attn_dropout,
                                   scale_attn_weights=True, training=False, attn_mask=None, head_mask=None)
            # out: [B * agent_num, 1, seq_len, 1] -> [B * agent_num, seq_len, 1]
            out = ops.merge_heads(out, num_heads, 1)
            ############## weight sum of time layer
            # out: [B * agent_num, seq_len, 1] -> [B, seq_len, agent_num, 1]
            output = out.reshape(batch_size, agent_num, seq_length, 1).permute(0, 2, 1, 3)
        else:
            # (batch_size, seq_length, agent_num, 1)
            output = self.last(time_output)

        return output


