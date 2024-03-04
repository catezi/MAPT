import math
import json
import torch
import numpy as np
import torch.nn as nn
from types import SimpleNamespace
from torch.nn import functional as F


def apply_activation(activation='linear'):
    if activation == 'linear':
        return nn.Identity()
    elif activation == 'none':
        return nn.Identity()
    # elif activation == 'gelu_new':
    #     return 0.5 * x * (1.0 + F.Tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    # elif activation == 'gelu_fast':
    #     return 0.5 * x * (1.0 + F.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))
    elif activation == 'gelu':
        return nn.GELU()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'leaky_relu':
        return nn.LeakyReLU()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'tanh':
        return nn.Tanh()
    else:
        raise ValueError(f'Unknown activation function: {activation}.')


##################################################################### TO be checked
def linear(features, param_dict, bias=True):
    if param_dict is None:
        return nn.Dense(features=features, use_bias=bias)
    else:
        if bias:
            assert 'bias' in param_dict
            assert 'weight' in param_dict
            return nn.Dense(features=features,
                            kernel_init=lambda *_: jnp.array(param_dict['weight']),
                            bias_init=lambda *_: jnp.array(param_dict['bias']))
        else:
            assert 'weight' in param_dict
            return nn.Dense(features=features,
                            kernel_init=lambda *_: jnp.array(param_dict['weight']))


def embedding(num_embeddings, features, param_dict, dtype='float32'):
    if param_dict is None:
        return nn.Embed(num_embeddings=num_embeddings, features=features, dtype=dtype)
    else:
        assert 'weight' in param_dict
        embedding_init = lambda *_: jnp.array(param_dict['weight'])
        return nn.Embed(num_embeddings=num_embeddings, features=features, embedding_init=embedding_init, dtype=dtype)


##################################################################### TO be checked


##################################################################### TO be checked
def layer_norm(param_dict, use_bias=True, use_scale=True, eps=1e-06, dtype='float32'):
    if param_dict is None:
        return nn.LayerNorm(use_bias=use_bias, use_scale=use_scale, epsilon=eps, dtype=dtype)
    else:
        kwargs = {'use_bias': use_bias, 'use_scale': use_scale, 'epsilon': eps, 'dtype': dtype}
        if use_bias:
            assert 'bias' in param_dict, 'use_bias is set True but bias parameter does not exist in param_dict.'
            kwargs['bias_init'] = lambda *_: jnp.array(param_dict['bias'])
        if use_scale:
            assert 'scale' in param_dict, 'use_scale is set True but scale parameter does not exist in param_dict.'
            kwargs['scale_init'] = lambda *_: jnp.array(param_dict['scale'])
        return nn.LayerNorm(**kwargs)


############################## method for attention
###### checked
def split_heads(x, num_heads, head_dim):
    """
    Splits embeddings for different heads.

    Args:
        x (tensor): Input tensor, shape [B, seq_len, embd_dim] or [B, blocks, block_len, embd_dim].
        num_heads (int): Number of heads.
        head_dim (int): Dimension of embedding for each head.

    Returns:
        (tensor): Output tensor, shape [B, num_head, seq_len, head_dim] or [B, blocks, num_head, block_len, head_dim].
    """
    newshape = x.shape[:-1] + (num_heads, head_dim)
    x = torch.reshape(x, newshape)
    if x.ndim == 5:
        # [batch, seq_len, agent_num, head, head_dim] =>
        # [batch, head, seq_len, agent_num, head_dim]
        return torch.permute(x, dims=(0, 3, 1, 2, 4))
    elif x.ndim == 4:
        # [batch, head, seq_len, head_dim]
        return torch.permute(x, dims=(0, 2, 1, 3))
    else:
        raise ValueError(f'Input tensor should have rank 4 or 5, but has rank {x.ndim}.')


###### checked
def merge_heads(x, num_heads, head_dim):
    """
    Merge embeddings for different heads.
    Args:
        x (tensor): Input tensor, shape [B, num_head, seq_len, head_dim] or [B, blocks, num_head, block_len, head_dim].
        num_heads (int): Number of heads.
        head_dim (int): Dimension of embedding for each head.

    Returns:
        (tensor): Output tensor, shape [B, seq_len, embd_dim] or [B, blocks, block_len, embd_dim].
    """
    if x.ndim == 5:
        x = torch.permute(x, dims=(0, 1, 3, 2, 4))
    elif x.ndim == 4:
        x = torch.permute(x, dims=(0, 2, 1, 3))
    else:
        raise ValueError(f'Input tensor should have rank 4 or 5, but has rank {x.ndim}.')

    newshape = x.shape[:-2] + (num_heads * head_dim,)
    x = torch.reshape(x, newshape)
    return x


###### checked
def attention(query, key, value, casual_mask, masked_bias, dropout,
              scale_attn_weights, training, attn_mask=None, head_mask=None):
    """
    Computes Dot-Product Attention for the given query, key and value.
    
    Args:
        query (tensor): Query, shape [B, num_heads, seq_len, embd_dim].
        key (tensor): Key, shape [B, num_heads, seq_len, embd_dim].
        value (tensor): Value, shape [B, num_heads, seq_len, embd_dim].
        casual_mask (tensor): Mask to ensure that attention is only applied to the left of the input sequence, 
                              shape [1, 1, key_len - query_len :key_len, :key_len].
        masked_bias (float): Value to insert for masked part of the sequence.
        dropout (nn.Dropout): Dropout module that is applied to the attention output.
        scale_attn_weights (bool): If True, scale the attention weights.
        training (bool): Training mode.
        attn_mask (tensor): Mask to avoid performing attention on padded tokens indices, shape [B, seq_len].
        head_mask (tensor): Mask to nullify selected heads of the self-attention modules, shape [num_heads,] or [num_layers, num_heads].
        feedback (tensor): external feedback with marked points.

    Returns:
        (tensor): Attention output, shape [B, num_heads, seq_len, embd_dim].
        (tensor): Attention weights, shape [B, num_heads, seq_len, seq_len].
        (tensor): KLD loss with external feedback, float.
    """
    query = query.to(dtype=torch.float32)
    key = key.to(dtype=torch.float32)
    attn_weights = torch.matmul(query, torch.swapaxes(key, -1, -2))

    if scale_attn_weights:
        attn_weights = attn_weights / (float(value.shape[-1]) ** 0.5)
    """
    attn_weights torch.Size([160, 4, 200, 200])
    casual_mask torch.Size([1, 1, 200, 200])
    attn_mask torch.Size([160, 1, 1, 200])
    """
    masked_bias = torch.tensor(masked_bias).to(device=attn_weights.device, dtype=torch.float32)
    attn_weights = torch.where(casual_mask, attn_weights, masked_bias)
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask

    _attn_weights = F.softmax(attn_weights, dim=-1)
    attn_weights = _attn_weights.to(dtype=value.dtype)
    attn_weights = dropout(attn_weights)

    if head_mask is not None:
        attn_weights = attn_weights * head_mask

    out = torch.matmul(attn_weights, value)
    return out, _attn_weights


###### checked
def get_attention_mask(attn_mask, batch_size):
    assert batch_size > 0, 'batch_size should be > 0.'
    attn_mask = torch.reshape(attn_mask, (batch_size, -1))
    attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
    attn_mask = (1.0 - attn_mask) * -10000.0
    return attn_mask


###### checked
def get_head_mask(head_mask, num_layers):
    if head_mask.ndim == 1:
        head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        head_mask = torch.repeat_interleave(head_mask, repeats=num_layers, dim=0)
    elif head_mask.ndim == 2:
        head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
    else:
        raise ValueError(f'head_mask must have rank 5, but has rank {head_mask.ndim}.')
    return head_mask


############################## method for loss
def cross_entropy(logits, labels, ignore_index=-100):
    """
    Computes the cross entroy loss (on logits).

    Args:
        logits (tensor): Logits, shape [B, num_classes].
        labels (tensor): Labels, shape [B,].
        ignore_index (int): Value of label to ignore for loss computation.

    Returns:
        (tensor): Cross entroy loss.
    """
    batch_size, num_classes = logits.shape
    logits = nn.log_softmax(logits)
    # Get indices where label is equal to ignore_index
    idx = jnp.nonzero(labels == ignore_index)[0]
    one_hot_labels = jax.nn.one_hot(labels, num_classes=num_classes)
    mult = one_hot_labels * logits
    # Insert zeros, where the labels are equal to ignore_index
    mult = mult.at[idx].set(jnp.zeros((idx.shape[0], num_classes)))
    return -jnp.sum(jnp.sum(mult, axis=-1)) / (batch_size - idx.shape[0])


def kld_loss(p, q):
    return jnp.sum(jnp.where(p != 0, p * (jnp.log(p) - jnp.log(q)), 0))


def custom_softmax(array, axis=-1, temperature=1.0):
    array = array / temperature
    return jax.nn.softmax(array, axis=axis)


def mse_loss(val, target):
    return jnp.mean(jnp.square(val - target))


# ----------------------------------------------------------
# Misc
# ----------------------------------------------------------
def get(dictionary, key):
    if dictionary is None or key not in dictionary:
        return None
    return dictionary[key]


def load_config(path):
    return json.loads(open(path, 'r', encoding='utf-8').read(), object_hook=lambda d: SimpleNamespace(**d))
