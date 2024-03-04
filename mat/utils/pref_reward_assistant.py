import os
import torch
import pickle
import numpy as np
import transformers
from collections import defaultdict
from mat.algorithms.reward_model.models.MR import MR
from mat.algorithms.reward_model.models.NMR import NMR
from mat.algorithms.reward_model.models.lstm import LSTMRewardModel
from mat.algorithms.reward_model.models.torch_utils import batch_to_torch
from mat.algorithms.reward_model.models.PrefTransformer import PrefTransformer
from mat.algorithms.reward_model.models.trajectory_gpt2 import TransRewardModel
from mat.algorithms.reward_model.models.q_function import FullyConnectedQFunction
from mat.utils.util import check, get_shape_from_obs_space, get_shape_from_act_space
from mat.algorithms.reward_model.models.MultiPrefTransformer import MultiPrefTransformer
from mat.algorithms.reward_model.models.encoder_decoder_divide import MultiTransRewardDivideModel


class RewardNorm:
    def __init__(self, shape=(), reward_std=1.0):
        self.reward_std = reward_std
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM)/self._n
            self._S[...] = self._S + (x - oldM)*(x - self._M)

    def norm(self, reward):
        return ((reward - self._M) / (self.std + 1e-12)) * self.reward_std

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        if self._n >= 2:
            return self._S/(self._n - 1)
        else:
            return np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


class PrefRewardAssistant:
    def __init__(self, args, obs_space, act_space, num_agents, device='cpu'):
        self.args = args
        self.episode_length = args.episode_length
        self.preference_eval = args.preference_eval
        self.n_rollout_threads = args.n_rollout_threads \
            if not args.preference_eval else args.n_eval_rollout_threads
        self.num_agents = num_agents
        self.device = device
        self.preference_model_type = args.preference_model_type
        self.preference_traj_length = args.preference_traj_length
        self.preference_agent_individual = args.preference_agent_individual
        self.preference_medium_process_type = args.preference_medium_process_type
        obs_shape = get_shape_from_obs_space(obs_space)
        act_shape = get_shape_from_act_space(act_space)
        if type(obs_shape[-1]) == list:
            obs_shape = obs_shape[:1]
        ######################### load max_traj_length from save model params
        model_para = torch.load(args.preference_model_dir)
        self.preference_traj_length = model_para['seq_len'] if 'seq_len' in model_para else self.preference_traj_length
        self.preference_traj_length = args.preference_traj_length if args.preference_config_length else self.preference_traj_length
        self.model_traj_length = model_para['seq_len'] if 'seq_len' in model_para else self.preference_traj_length
        print('self.preference_traj_length', self.preference_traj_length)
        print('self.model_traj_length', self.model_traj_length)
        ######################### history data for preference reward
        self.his_obs = np.zeros((
            self.n_rollout_threads, self.preference_traj_length, num_agents, *obs_shape), dtype=np.float32)
        self.his_next_obs = np.zeros((
            self.n_rollout_threads, self.preference_traj_length, num_agents, *obs_shape), dtype=np.float32)
        self.his_act = np.zeros((
            self.n_rollout_threads, self.preference_traj_length, num_agents, act_shape), dtype=np.float32)
        self.his_timesteps = np.zeros((
            self.n_rollout_threads, self.preference_traj_length), dtype=np.long)
        self.his_attn_masks = np.zeros((
            self.n_rollout_threads, self.preference_traj_length, num_agents), dtype=np.float32)
        self.last_dones = np.zeros(self.n_rollout_threads, dtype=np.float32)
        ######################### config prefreence reward model
        action_type = 'Continous' if (args.env_name == 'hands' or args.env_name == 'mujoco') else 'Discrete'
        if args.preference_model_type == "MultiPrefTransformerDivide":
            trans_config = MultiPrefTransformer.get_default_config()
            update_multi_trans_config(trans_config, args)
            config = transformers.GPT2Config(**trans_config)
            trans = MultiTransRewardDivideModel(
                config=config, observation_dim=self.his_obs.shape[-1],
                action_dim=act_space.n if action_type == 'Discrete' else act_shape,
                n_agent=self.num_agents, action_type=action_type,
                max_episode_steps=self.model_traj_length, device=device,
            )
            # config model wrapper for train and eval
            self.reward_model = MultiPrefTransformer(config, trans, device)
        elif args.preference_model_type == 'PrefTransformer':
            trans_config = PrefTransformer.get_default_config()
            update_trans_config(trans_config, args)
            config = transformers.GPT2Config(**trans_config)
            trans = TransRewardModel(
                config=config, observation_dim=self.his_obs.shape[-1],
                action_dim=act_space.n if action_type == 'Discrete' else act_shape,
                action_type=action_type, activation='relu', activation_final='none',
                max_episode_steps=self.model_traj_length, device=device)
            # config model wrapper for train and eval
            self.reward_model = PrefTransformer(config, trans, device)
        elif args.preference_model_type == "MR":
            reward_config = MR.get_default_config()
            update_reward_config(reward_config, args)
            rf = FullyConnectedQFunction(
                observation_dim=self.his_obs.shape[-1], action_dim=act_space.n if action_type == 'Discrete' else act_shape,
                action_type=action_type, inner_dim=args.preference_reward_inner_dim, orthogonal_init=args.preference_reward_orthogonal_init,
                activations='relu', activation_final='none', device=device,
            )
            self.reward_model = MR(reward_config, rf, device)
        elif args.preference_model_type == "NMR":
            lstm_config = NMR.get_default_config()
            update_lstm_config(lstm_config, args)
            config = transformers.GPT2Config(**lstm_config)
            lstm = LSTMRewardModel(
                config=config, observation_dim=self.his_obs.shape[-1],
                action_dim=act_space.n if action_type == 'Discrete' else act_shape,
                action_type=action_type, activation='relu', activation_final='none',
                max_episode_steps=self.model_traj_length, device=device,
            )
            self.reward_model = NMR(config, lstm, device)
        else:
            raise NotImplementedError()
        self.reward_model.load_model(args.preference_model_dir)
        ######################### normalize preference reward if necessary
        self.pref_reward_norm = RewardNorm(
            shape=(1, ), reward_std=args.preference_reward_std
        ) if self.args.preference_reward_norm else None

    def insert(self, obs, actions, dones):
        ######################### upadte obs
        self.his_obs = self.his_next_obs.copy()
        ######################### upadte action
        self.his_act[:, 0: self.preference_traj_length - 1] = self.his_act[:, 1: self.preference_traj_length]
        self.his_act[:, self.preference_traj_length - 1] = actions.copy().astype(np.float32)
        ######################### upadte timesteps
        self.his_timesteps[:, 0: self.preference_traj_length - 1] = self.his_timesteps[:, 1: self.preference_traj_length]
        self.his_timesteps[:, self.preference_traj_length - 1] = self.his_timesteps[:, self.preference_traj_length - 2] + 1
        for i in range(self.his_timesteps.shape[0]):
            if np.max(self.his_timesteps[i]) >= self.preference_traj_length:
                self.his_timesteps[i] = np.arange(0, self.preference_traj_length)
        ######################### upadte atten_masks
        self.his_attn_masks[:, 0: self.preference_traj_length - 1] = self.his_attn_masks[:, 1: self.preference_traj_length]
        self.his_attn_masks[:, self.preference_traj_length - 1] = 1
        ######################### upadte next obs
        self.his_next_obs[:, 0: self.preference_traj_length - 1] = self.his_next_obs[:, 1: self.preference_traj_length]
        self.his_next_obs[:, self.preference_traj_length - 1] = obs.copy()
        # if np.all(self.last_dones):
        #     print('----------- insert data -----------')
        #     # print('self.his_obs', self.his_obs.shape)
        #     # print('self.his_act', self.his_act.shape)
        #     # print('self.his_timesteps', self.his_timesteps.shape)
        #     # print('self.his_attn_masks', self.his_attn_masks.shape)
        #     print('self.his_obs', np.mean(self.his_obs[0, :, 0], axis=-1))
        #     print('self.his_next_obs', np.mean(self.his_next_obs[0, :, 0], axis=-1))
        #     print('self.his_act', np.mean(self.his_act[0, :, 0], axis=-1))
        #     print('self.his_timesteps', self.his_timesteps[0, :])
        #     print('self.his_attn_masks', self.his_attn_masks[0, :, 0])
        ######################### get preference reward from reward model
        if self.preference_model_type == 'MR':  # MR only need step obs, action for pref
            torch_input = batch_to_torch(dict(
                observations=self.his_obs[:, -1],
                actions=self.his_act[:, -1],
                timestep=self.his_timesteps[:, -1],
                attn_mask=self.his_attn_masks[:, -1],
            ), self.device)
        else:
            torch_input = batch_to_torch(dict(
                observations=self.his_obs,
                actions=self.his_act,
                timestep=self.his_timesteps,
                attn_mask=self.his_attn_masks,
            ), self.device)
        pref_reward = self.reward_model.get_reward(torch_input)
        # MR and MPT have no timestep-dim
        if self.preference_model_type == 'MR':
            pref_reward = pref_reward.detach().cpu().numpy()
        elif self.preference_model_type == 'MultiPrefTransformer' and self.preference_medium_process_type != 'cat':
            pref_reward = pref_reward.detach().cpu().numpy()
        else:  # get last value from sequence
            pref_reward = pref_reward[:, -1].detach().cpu().numpy()
        ######################### replace last done
        self.last_dones = np.all(dones, axis=1).astype(self.last_dones.dtype)
        ######################### mean reward if user global reward
        if not self.preference_agent_individual:
            pref_reward = np.mean(pref_reward, axis=1)
            pref_reward = np.tile(pref_reward.reshape((self.n_rollout_threads, 1, 1)), (1, self.num_agents, 1))
        if self.args.preference_reward_norm and not self.preference_eval:
            for p_r in pref_reward.reshape(-1, 1):
                self.pref_reward_norm.push(p_r)
            pref_reward = self.pref_reward_norm.norm(pref_reward)
        return pref_reward

    def clear_done(self):
        # self.his_obs: (n_rollout_threads, traj_length, num_agents, obs_dim, )
        self.his_obs = self.his_obs * np.tile(
            (1 - self.last_dones).reshape(-1, 1, 1, 1), (1, *self.his_obs.shape[1:]))
        # self.his_act: (n_rollout_threads, traj_length, num_agents, act_dim, )
        self.his_act = self.his_act * np.tile(
            (1 - self.last_dones).reshape(-1, 1, 1, 1), (1, *self.his_act.shape[1:]))
        # self.his_timesteps: (n_rollout_threads, traj_length, )
        self.his_timesteps = self.his_timesteps * np.tile(
            (1 - self.last_dones.astype(np.long)).reshape(-1, 1), (1, *self.his_timesteps.shape[1:]))
        # self.his_timesteps: (n_rollout_threads, traj_length, num_agents, )
        self.his_attn_masks = self.his_attn_masks * np.tile(
            (1 - self.last_dones).reshape(-1, 1, 1), (1, *self.his_attn_masks.shape[1:]))
        # self.his_next_obs: (n_rollout_threads, traj_length, num_agents, act_dim)
        start_obs = self.his_next_obs[:, -1]
        for i in range(self.his_next_obs.shape[0]):
            if self.last_dones[i]:
                self.his_next_obs[i] = np.zeros_like(self.his_next_obs[i])
                self.his_next_obs[i, -1] = start_obs[-1]


def update_trans_config(config, user_config):
    config['use_weighted_sum'] = user_config.preference_use_weighted_sum
    config['agent_individual'] = user_config.preference_agent_individual
    config['embd_dim'] = user_config.preference_embd_dim
    config['pref_attn_embd_dim'] = user_config.preference_embd_dim
    config['n_layer'] = user_config.preference_n_layer
    config['n_head'] = user_config.preference_n_head
    config['reverse_state_action'] = user_config.preference_reverse_state_action


def update_reward_config(config, user_config):
    config['agent_individual'] = user_config.preference_agent_individual
    config['inner_dim'] = user_config.preference_reward_inner_dim
    config['action_embd_dim'] = user_config.preference_action_embd_dim
    config['orthogonal_init'] = user_config.preference_reward_orthogonal_init


def update_lstm_config(config, user_config):
    config['embd_dim'] = user_config.preference_embd_dim
    config['action_embd_dim'] = user_config.preference_action_embd_dim


def update_multi_trans_config(config, user_config):
    config['preference_reward_mean_agent'] = user_config.preference_reward_mean_agent
    config['embd_dim'] = user_config.preference_embd_dim
    config['n_layer'] = user_config.preference_n_layer
    config['n_head'] = user_config.preference_n_head
    config['reverse_state_action'] = user_config.preference_reverse_state_action
    config['medium_process_type'] = user_config.preference_medium_process_type
    config['use_dropout'] = user_config.preference_use_dropout
    config['use_lstm'] = user_config.preference_use_lstm
    config['add_obs_action'] = user_config.preference_add_obs_action
    config['drop_agent_layer'] = user_config.preference_drop_agent_layer
    config['use_highway'] = user_config.preference_use_highway
    config['attention_agent_first'] = user_config.preference_attention_agent_first
    config['encoder_mlp'] = user_config.preference_encoder_mlp
    config['decoder_mlp'] = user_config.preference_decoder_mlp
    config['agent_layer_mlp'] = user_config.preference_agent_layer_mlp
    config['time_layer_mlp'] = user_config.preference_time_layer_mlp


class RewardCollector:
    def __init__(self, args):
        self.args = args
        self.n_eval_rollout_threads = args.n_eval_rollout_threads
        self.all_real_rewards = [[] for _ in range(self.n_eval_rollout_threads)]
        self.all_pref_rewards = [[] for _ in range(self.n_eval_rollout_threads)]
        self.rewards_traj_buffer = []

    def insert(self, real_rewards, pref_rewards, dones):
        for i in range(self.n_eval_rollout_threads):
            self.all_real_rewards[i].append(real_rewards[i:i+1])
            self.all_pref_rewards[i].append(pref_rewards[i:i+1])
            if dones[i]:
                self.rewards_traj_buffer.append({
                    'real_rewards': np.concatenate(self.all_real_rewards[i], axis=0),
                    'pref_rewards': np.concatenate(self.all_pref_rewards[i], axis=0),
                })
                print('traj_real_rewards', np.concatenate(self.all_real_rewards[i], axis=0).shape)
                print('traj_pref_rewards', np.concatenate(self.all_pref_rewards[i], axis=0).shape)
                self.all_real_rewards[i] = []
                self.all_pref_rewards[i] = []

    def save_data(self):
        if not os.path.exists(self.args.preference_res_dir):
            os.makedirs(self.args.preference_res_dir)
        with open(self.args.preference_res_dir + '/rewards_data.pkl', 'wb') as f:
            pickle.dump(self.rewards_traj_buffer, f)
