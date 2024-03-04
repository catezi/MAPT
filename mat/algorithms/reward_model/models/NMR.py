import torch
import numpy as np
from functools import partial
from ml_collections import ConfigDict
from mat.algorithms.reward_model.models.torch_utils import mse_loss, cross_ent_loss


class NMR(object):

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.lstm_lr = 1e-3
        config.optimizer_type = 'adam'
        config.scheduler_type = 'none'
        config.vocab_size = 1
        config.n_layer = 3
        config.embd_dim = 256
        config.action_embd_dim = 64
        config.n_head = 1
        config.n_positions = 1024
        config.resid_pdrop = 0.1
        config.attn_pdrop = 0.1
        config.use_kld = False
        config.lambda_kld = 0.1
        config.softmax_temperature = 5
        config.train_type = "sum"
        config.train_diff_bool = False
        config.explicit_sparse = False
        config.agent_individual = False
        config.k = 5
        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, lstm, device):
        ####################### config basic info
        self.config = config
        self.lstm = lstm
        self.device = device
        self.observation_dim = lstm.observation_dim
        self.action_dim = lstm.action_dim
        optimizer_class = {
            'adam': torch.optim.Adam,
            'adamw': torch.optim.AdamW,
            'sgd': torch.optim.SGD,
        }[self.config.optimizer_type]
        self.optimizer = optimizer_class(self.lstm.parameters(), lr=self.config.lstm_lr)
        ####################### MR not set scheduler
        self._total_steps = 0

    def train(self, batch):
        self.lstm.train()
        self._total_steps += 1
        metrics = self._train_pref_step(batch)
        return metrics

    def _train_pref_step(self, batch):
        def loss_fn():
            obs_0 = batch['observations0']
            act_0 = batch['actions0']
            obs_1 = batch['observations1']
            act_1 = batch['actions1']
            labels = batch['labels']
            B, T, N, _ = batch['observations0'].shape
            B, T, N, _ = batch['actions0'].shape
            ####################### copmpute loss
            lstm_pred_0, _ = self.lstm(obs_0, act_0)
            lstm_pred_1, _ = self.lstm(obs_1, act_1)
            ####################### add all agents rewards as global reward(or individual reward)
            if self.config.agent_individual:
                lstm_pred_0 = lstm_pred_0.permute(0, 2, 1, 3).reshape(B * N, T, -1)
                lstm_pred_1 = lstm_pred_1.permute(0, 2, 1, 3).reshape(B * N, T, -1)
                labels = labels.unsqueeze(1).repeat(1, N, 1).reshape(B * N, -1)
                B = B * N
            else:
                lstm_pred_0 = torch.sum(lstm_pred_0, dim=-2)
                lstm_pred_1 = torch.sum(lstm_pred_1, dim=-2)
            if self.config.train_type == "mean":
                sum_pred_0 = torch.mean(lstm_pred_0.reshape(B, T), dim=1).reshape(-1, 1)
                sum_pred_1 = torch.mean(lstm_pred_1.reshape(B, T), dim=1).reshape(-1, 1)
            if self.config.train_type == "sum":
                sum_pred_0 = torch.sum(lstm_pred_0.reshape(B, T), dim=1).reshape(-1, 1)
                sum_pred_1 = torch.sum(lstm_pred_1.reshape(B, T), dim=1).reshape(-1, 1)
            elif self.config.train_type == "last":
                sum_pred_0 = lstm_pred_0.reshape(B, T)[:, -1].reshape(-1, 1)
                sum_pred_1 = lstm_pred_1.reshape(B, T)[:, -1].reshape(-1, 1)
            logits = torch.cat([sum_pred_0, sum_pred_1], dim=1)
            loss_collection = {}
            lstm_loss = cross_ent_loss(logits, labels)
            ####################### copmpute grad and update model
            self.optimizer.zero_grad()
            lstm_loss.backward()
            self.optimizer.step()
            loss_collection['lstm_loss'] = lstm_loss.detach().cpu().numpy()
            return loss_collection
        aux_values = loss_fn()
        metrics = dict(
            lstm_loss=aux_values['lstm_loss'],
        )
        return metrics

    def evaluation(self, batch):
        self.lstm.eval()
        metrics = self._eval_pref_step(batch)
        return metrics

    def _eval_pref_step(self, batch):
        def loss_fn():
            obs_0 = batch['observations0']
            act_0 = batch['actions0']
            obs_1 = batch['observations1']
            act_1 = batch['actions1']
            labels = batch['labels']
            B, T, N, _ = batch['observations0'].shape
            B, T, N, _ = batch['actions0'].shape
            ####################### copmpute loss
            lstm_pred_0, _ = self.lstm(obs_0, act_0)
            lstm_pred_1, _ = self.lstm(obs_1, act_1)
            ####################### add all agents rewards as global reward(or individual reward)
            if self.config.agent_individual:
                lstm_pred_0 = lstm_pred_0.permute(0, 2, 1, 3).reshape(B * N, T, -1)
                lstm_pred_1 = lstm_pred_1.permute(0, 2, 1, 3).reshape(B * N, T, -1)
                labels = labels.unsqueeze(1).repeat(1, N, 1).reshape(B * N, -1)
                B = B * N
            else:
                lstm_pred_0 = torch.sum(lstm_pred_0, dim=-2)
                lstm_pred_1 = torch.sum(lstm_pred_1, dim=-2)
            if self.config.train_type == "mean":
                sum_pred_0 = torch.mean(lstm_pred_0.reshape(B, T), dim=1).reshape(-1, 1)
                sum_pred_1 = torch.mean(lstm_pred_1.reshape(B, T), dim=1).reshape(-1, 1)
            if self.config.train_type == "sum":
                sum_pred_0 = torch.sum(lstm_pred_0.reshape(B, T), dim=1).reshape(-1, 1)
                sum_pred_1 = torch.sum(lstm_pred_1.reshape(B, T), dim=1).reshape(-1, 1)
            elif self.config.train_type == "last":
                sum_pred_0 = lstm_pred_0.reshape(B, T)[:, -1].reshape(-1, 1)
                sum_pred_1 = lstm_pred_1.reshape(B, T)[:, -1].reshape(-1, 1)
            logits = torch.cat([sum_pred_0, sum_pred_1], dim=1)
            loss_collection = {}
            lstm_loss = cross_ent_loss(logits, labels)
            loss_collection['lstm_loss'] = lstm_loss.detach().cpu().numpy()
            return loss_collection
        aux_values = loss_fn()
        metrics = dict(
            eval_lstm_loss=aux_values['lstm_loss'],
        )
        return metrics

    ##################################### TO be checked
    def get_reward(self, batch):
        self.lstm.eval()
        return self._get_reward_step(batch)

    def _get_reward_step(self, batch):
        obs = batch['observations']
        act = batch['actions']
        lstm_pred, _ = self.lstm(obs, act)
        return lstm_pred
    ##################################### TO be checked

    ####################### my add method
    def save_model(self, save_path, save_idx):
        torch.save({
            'reward_model': self.lstm.state_dict(),
            'seq_len': self.lstm.max_episode_steps,
        }, str(save_path) + "reward_model_" + str(save_idx) + ".pt")

    def load_model(self, model_dir):
        model_state_dict = torch.load(model_dir, map_location=torch.device('cpu')) \
            if self.device == torch.device('cpu') else torch.load(model_dir)
        self.lstm.load_state_dict(model_state_dict['reward_model'])
        print('--------------- load NMR -----------------')

    @property
    def total_steps(self):
        return self._total_steps
