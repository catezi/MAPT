import torch
import numpy as np
import torch.nn as nn
import mat.algorithms.reward_model.models.ops as ops
from torch.nn import functional as F
from typing import Any

URLS = {'gpt2': 'https://www.dropbox.com/s/0wdgj0gazwt9nm7/gpt2.h5?dl=1',
        'gpt2-medium': 'https://www.dropbox.com/s/nam11kbd83wsm7d/gpt2-medium.h5?dl=1',
        'gpt2-large': 'https://www.dropbox.com/s/oy8623qwkkjm8gt/gpt2-large.h5?dl=1',
        'gpt2-xl': 'https://www.dropbox.com/s/6c6qt0bzz4v2afx/gpt2-xl.h5?dl=1'}

CONFIGS = {'gpt2': 'https://www.dropbox.com/s/s5xl32dgwc8322p/gpt2.json?dl=1',
           'gpt2-medium': 'https://www.dropbox.com/s/7mwkijxoh1earm5/gpt2-medium.json?dl=1',
           'gpt2-large': 'https://www.dropbox.com/s/nhslkxwxtpn7auz/gpt2-large.json?dl=1',
           'gpt2-xl': 'https://www.dropbox.com/s/1iv0nq1xigsfdvb/gpt2-xl.json?dl=1'}


class GPT2SelfAttention(nn.Module):
    """
    GPT2 Self Attention.
    Attributes:
        config (Any): Configuration object. If 'pretrained' is not None, this parameter will be ignored.
        param_dict (dict): Parameter dict with pretrained parameters. If not None, 'pretrained' will be ignored.
    """
    def __init__(self, config, device):
        super(GPT2SelfAttention, self).__init__()
        ####################### set basic config
        self.config = config
        self.device = device
        self.max_pos = self.config.n_positions
        self.embd_dim = self.config.embd_dim
        self.num_heads = self.config.n_head
        self.head_dim = self.embd_dim // self.num_heads
        self.attn_dropout = self.config.attn_pdrop
        self.resid_dropout = self.config.resid_pdrop
        self.scale_attn_weights = True
        ####################### set model structure
        self.qkv = nn.Linear(self.embd_dim, 3 * self.embd_dim)
        self.att_drop = nn.Dropout(self.attn_dropout)
        self.proj = nn.Linear(self.embd_dim, self.embd_dim)
        self.resid_drop = nn.Dropout(self.resid_dropout)

    def forward(self, x, layer_past=None, attn_mask=None, head_mask=None, use_cache=False, training=False):
        x = self.qkv(x)
        qkv = torch.split(x, self.embd_dim, dim=-1)
        query, key, value = qkv[0], qkv[1], qkv[2]
        query = ops.split_heads(query, self.num_heads, self.head_dim)
        value = ops.split_heads(value, self.num_heads, self.head_dim)
        key = ops.split_heads(key, self.num_heads, self.head_dim)
        ##################################################################### TO be checked
        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        present = (key, value) if use_cache else None
        query_len, key_len = query.shape[-2], key.shape[-2]
        casual_mask = torch.tril(torch.ones(
            (1, 1, self.max_pos, self.max_pos), device=self.device))[:, :, key_len - query_len:key_len, :key_len]
        # casual_mask = torch.ones(
        #     (1, 1, self.max_pos, self.max_pos), device=self.device)[:, :, key_len - query_len :key_len, :key_len]
        casual_mask = casual_mask.to(dtype=torch.bool)
        out, _attn_weights = ops.attention(query, key, value, casual_mask, -1e4, self.att_drop,
                                           self.scale_attn_weights, training, attn_mask, head_mask)
        out = ops.merge_heads(out, self.num_heads, self.head_dim)
        out = self.proj(out)
        out = self.resid_drop(out)
        return out, present, _attn_weights


class GPT2MLP(nn.Module):
    """
    GPT2 MLP.
    Attributes:
        intermediate_dim (int): Dimension of the intermediate layer.
        config (Any): Configuration object. If 'pretrained' is not None, this parameter will be ignored.
        param_dict (dict): Parameter dict with pretrained parameters. If not None, 'pretrained' will be ignored.
    """
    def __init__(self, intermediate_dim, config):
        super(GPT2MLP, self).__init__()
        ####################### set basic config
        self.intermediate_dim = intermediate_dim
        self.config = config
        self.embd_dim = self.config.embd_dim
        self.resid_dropout = self.config.resid_pdrop
        self.activation = self.config.activation_function
        ####################### set model structure
        self.mlp = nn.Sequential(
            nn.Linear(self.embd_dim, self.intermediate_dim),
            ops.apply_activation(self.activation),
            nn.Linear(self.intermediate_dim, self.embd_dim),
            nn.Dropout(self.resid_dropout),
        )

    def forward(self, x, training=False):
        x = self.mlp(x)
        return x


class GPT2Block(nn.Module):
    """
    GPT2 Block.
    Attributes:
        config (Any): Configuration object. If 'pretrained' is not None, this parameter will be ignored.
        param_dict (dict): Parameter dict with pretrained parameters. If not None, 'pretrained' will be ignored.
    """
    def __init__(self, config, device):
        super(GPT2Block, self).__init__()
        ####################### set basic config
        self.config = config
        self.device = device
        self.embd_dim = self.config.embd_dim
        self.eps = self.config.layer_norm_epsilon
        self.inner_dim = self.config.n_inner if self.config.n_inner is not None else 4 * self.embd_dim
        ####################### set model structure
        self.ln1 = nn.LayerNorm(self.embd_dim, eps=self.eps)
        self.attention = GPT2SelfAttention(config=self.config, device=device)
        self.ln2 = nn.LayerNorm(self.embd_dim, eps=self.eps)
        self.ffn = GPT2MLP(intermediate_dim=self.inner_dim, config=self.config)

    def forward(self, x, layer_past=None, attn_mask=None, head_mask=None, use_cache=False, training=False):
        residual = x
        x = self.ln1(x)
        kwargs = {'layer_past': layer_past, 'attn_mask': attn_mask, 'head_mask': head_mask,
                  'use_cache': use_cache, 'training': training}
        x, present, _attn_weights = self.attention(x, **kwargs)
        x += residual
        residual = x
        x = self.ln2(x)
        x = self.ffn(x, training)
        x += residual
        return x, present, _attn_weights


class GPT2Model(nn.Module):
    """
    The GPT2 Model.
    Attributes:
        config (Any): Configuration object. If 'pretrained' is not None, this parameter will be ignored.
        ckpt_dir (str): Directory to which the pretrained weights are downloaded. If None, a temp directory will be used.
        param_dict (dict): Parameter dict with pretrained parameters. If not None, 'pretrained' will be ignored.
    """

    def __init__(self, config, device):
        super(GPT2Model, self).__init__()
        ####################### set basic config
        self.config_ = config
        self.device = device
        self.vocab_size = self.config_.vocab_size
        self.max_pos = self.config_.n_positions
        self.embd_dim = self.config_.embd_dim
        self.embd_dropout = self.config_.embd_pdrop
        self.num_layers = self.config_.n_layer
        self.eps = self.config_.layer_norm_epsilon
        ####################### set model structure
        self.drop = nn.Dropout(self.embd_dropout)
        self.blocks = [GPT2Block(config=self.config_, device=self.device).to(device) for _ in range(self.num_layers)]
        self.ln = nn.LayerNorm(self.embd_dim, eps=self.eps)

    def forward(self, input_ids=None, past_key_values=None, input_embds=None,
                attn_mask=None, head_mask=None, use_cache=False, training=False):
        ####################### process info to compute attention
        if input_ids is not None and input_embds is not None:
            raise ValueError('You cannot specify both input_ids and input_embd at the same time.')
        elif input_ids is not None:
            input_shape = input_ids.shape
            input_ids = torch.reshape(input_ids, (-1, input_shape[-1]))
            batch_size = input_ids.shape[0]
        elif input_embds is not None:
            batch_size = input_embds.shape[0]
        else:
            raise ValueError('You have to specify either input_ids or input_embd.')
        if past_key_values is None:
            past_key_values = tuple([None] * self.num_layers)
        if input_embds is None:
            input_embds = nn.Embed(num_embeddings=self.vocab_size, features=self.embd_dim)(input_ids)
        if attn_mask is not None:
            attn_mask = ops.get_attention_mask(attn_mask, batch_size)
        if head_mask is not None:
            head_mask = ops.get_head_mask(head_mask, self.num_layers)
        else:
            head_mask = [None] * self.num_layers
        ####################### start compute attention
        x = input_embds
        x = self.drop(x)
        presents = () if use_cache else None
        attn_weights_list = []
        for i in range(self.num_layers):
            kwargs = {'layer_past': past_key_values[i], 'attn_mask': attn_mask, 'head_mask': head_mask[i],
                      'use_cache': use_cache, 'training': training}
            x, present, attn_weights = self.blocks[i](x, **kwargs)
            if use_cache:
                presents = presents + (present,)
            attn_weights_list.append(attn_weights)
        x = self.ln(x)
        return {'last_hidden_state': x, 'past_key_values': presents, 'attn_weights_list': attn_weights_list}


class TransRewardModel(nn.Module):

    def __init__(self, config, observation_dim, action_dim, action_type, activation, activation_final,
                 pretrained=None, ckpt_dir=None, max_episode_steps=1000, device='cpu'):
        super(TransRewardModel, self).__init__()
        ####################### set basic config
        # set config form params
        self.config_ = config
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.action_type = action_type
        self.activation = activation
        self.activation_final = activation_final
        self.device = device
        # set other config (base on origin setup)
        self.config_.activation_function = self.activation
        self.config_.activation_final = self.activation_final
        self.vocab_size = self.config_.vocab_size
        self.max_pos = self.config_.n_positions
        self.embd_dim = self.config_.embd_dim
        self.pref_attn_embd_dim = self.config_.pref_attn_embd_dim
        self.embd_dropout = self.config_.embd_pdrop
        self.attn_dropout = self.config_.attn_pdrop
        self.resid_dropout = self.config_.resid_pdrop
        self.num_layers = self.config_.n_layer
        self.inner_dim = self.config_.embd_dim // 2
        self.eps = self.config_.layer_norm_epsilon
        self.max_episode_steps = max_episode_steps
        ####################### set model structure
        self.embed_state = nn.Linear(self.observation_dim, self.embd_dim)
        self.embed_action = nn.Linear(self.action_dim, self.embd_dim)
        self.embed_timestep = nn.Embedding(self.max_episode_steps + 1, self.embd_dim)
        self.embed_ln = nn.LayerNorm(self.embd_dim, eps=self.eps)
        self.gpt2_model = GPT2Model(config=self.config_, device=self.device)
        if self.config_.use_weighted_sum:
            self.header = nn.Linear(self.embd_dim, 2 * self.pref_attn_embd_dim + 1)
            self.attn_dropout = nn.Dropout(0.0)
        else:
            self.header = nn.Sequential(
                nn.Linear(self.embd_dim, self.inner_dim),
                ops.apply_activation(self.activation),
                nn.Linear(self.inner_dim, 1),
                ops.apply_activation(self.activation_final),
            )
        ####################### copy mmodel to correct device
        self.to(device)

    def forward(self, states, actions, timesteps, attn_mask=None, training=False, reverse=False, target_idx=1):
        batch_size, seq_length, agent_num = states.shape[0], states.shape[1], states.shape[2]
        if attn_mask is None:
            attn_mask = torch.ones((batch_size, seq_length, agent_num), dtype=torch.float32, device=self.device)
        # print('-------------------------')
        # print('states', states.shape)
        # print('actions', actions.shape)
        # print('timesteps', timesteps.shape)
        # print('attn_mask', attn_mask.shape)
        ####################### transform discrete action to onehot
        if self.action_type == 'Discrete':
            actions = F.one_hot(actions.squeeze(-1).long(), num_classes=self.action_dim).float()
        ####################### compute states, actions, timesteps embedding
        # use mean to all agents states, actions
        embd_state = self.embed_state(states)
        embd_action = self.embed_action(actions)
        embd_timestep = self.embed_timestep(timesteps).unsqueeze(2).repeat(1, 1, agent_num, 1)
        embd_state = embd_state + embd_timestep
        embd_action = embd_action + embd_timestep
        if not self.config_.reverse_state_action:
            stacked_inputs = torch.stack([
                embd_state, embd_action], dim=1).permute(0, 2, 1, 3, 4).reshape(batch_size, 2 * seq_length, agent_num, self.embd_dim)
        else:
            stacked_inputs = torch.stack([
                embd_action, embd_state], dim=1).permute(0, 2, 1, 3, 4).reshape(batch_size, 2 * seq_length, agent_num, self.embd_dim)
        stacked_inputs = stacked_inputs.permute(0, 2, 1, 3).reshape(batch_size * agent_num, 2 * seq_length, self.embd_dim)
        stacked_inputs = self.embed_ln(stacked_inputs)
        stacked_attn_mask = torch.stack([
            attn_mask, attn_mask], dim=1).permute(0, 2, 1, 3).reshape(batch_size, 2 * seq_length, agent_num)
        stacked_attn_mask = stacked_attn_mask.permute(0, 2, 1).reshape(batch_size * agent_num, 2 * seq_length)
        ####################### calculate first causal attention
        transformer_outputs = self.gpt2_model(
            input_embds=stacked_inputs,
            attn_mask=stacked_attn_mask,
            training=training,
        )
        x = transformer_outputs["last_hidden_state"]
        attn_weights_list = transformer_outputs["attn_weights_list"]
        # x = x.reshape(batch_size, agent_num, seq_length * 2, self.embd_dim).permute(0, 2, 1, 3)
        x = x.reshape(batch_size * agent_num, seq_length, 2, self.embd_dim).permute(0, 2, 1, 3)
        hidden_output = x[:, target_idx]
        ####################### calculate second preference attention
        # if self.config_.use_weighted_sum and training:
        if self.config_.use_weighted_sum:
            '''
            add additional Attention Layer for Weighted Sum.
            x (= output, tensor): Predicted Reward, shape [B, seq_len, embd_dim]
            '''
            x = self.header(hidden_output)
            # only one head, because value has 1 dim for predicting rewards directly.
            num_heads = 1
            # query: [B, seq_len, embd_dim]
            # key: [B, seq_len, embd_dim]
            # value: [B, seq_len, 1]
            qkv = torch.split(x, [self.pref_attn_embd_dim, self.pref_attn_embd_dim, 1], dim=2)
            query, key, value = qkv[0], qkv[1], qkv[2]
            query = ops.split_heads(query, num_heads, self.pref_attn_embd_dim)
            key = ops.split_heads(key, num_heads, self.pref_attn_embd_dim)
            value = ops.split_heads(value, num_heads, 1)
            # query: [B, 1, seq_len, embd_dim]
            # key: [B, 1, seq_len, embd_dim]
            # value: [B, 1, seq_len, 1]
            query_len, key_len = query.shape[-2], key.shape[-2]
            # casual_mask = jnp.tril(jnp.ones((1, 1, self.config_.n_positions, self.config_.n_positions)))[:, :, key_len - query_len :key_len, :key_len]
            # casual_mask = casual_mask.astype(bool)
            casual_mask = torch.ones((1, 1, seq_length, seq_length), device=self.device)[:, :, key_len - query_len:key_len, :key_len]
            casual_mask = casual_mask.to(dtype=torch.bool)
            new_attn_mask = ops.get_attention_mask(attn_mask, batch_size * agent_num)
            out, last_attn_weights = ops.attention(query, key, value, casual_mask, -1e-4, self.attn_dropout,
                                                   scale_attn_weights=True, training=training, attn_mask=new_attn_mask, head_mask=None)
            attn_weights_list.append(last_attn_weights)
            # out: [B * N, 1, seq_len, 1] -> output: [B * N, seq_len, 1]
            output = ops.merge_heads(out, num_heads, 1)
            # output: [B, N, seq_len, 1] -> [B, seq_len, N, 1]
            output = output.reshape(batch_size, agent_num, seq_length, 1).permute(0, 2, 1, 3)
            # value: [B * N, 1, seq_len, 1] -> [B * N, seq_len, 1]
            value = ops.merge_heads(value, num_heads, 1)
            # value: [B * N, seq_len, 1] -> [B, seq_len, N, 1]
            value = value.reshape(batch_size, agent_num, seq_length, 1).permute(0, 2, 1, 3)
            # output: [B, seq_len, 1]
            # output = nn.Dropout(rate=self.resid_dropout)(out, deterministic=not training)d
            return {"weighted_sum": output, "value": value}, attn_weights_list
        else:
            output = self.header(hidden_output)
            output = output.reshape(batch_size, agent_num, seq_length, 1).permute(0, 2, 1, 3)
            return {"value": output}, attn_weights_list

