import torch
import numpy as np
from ml_collections import ConfigDict
# from algorithms.reward_model.models.torch_utils import cross_ent_loss
from mat.algorithms.reward_model.models.torch_utils import cross_ent_loss


class PrefTransformer(object):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.trans_lr = 1e-4
        config.optimizer_type = 'adamw'
        config.scheduler_type = 'CosineDecay'
        config.vocab_size = 1
        config.n_layer = 1
        config.embd_dim = 256
        config.n_head = 4
        config.n_positions = 1024
        config.resid_pdrop = 0.1
        config.attn_pdrop = 0.1
        config.pref_attn_embd_dim = 256
        config.train_type = "mean"
        # Weighted Sum option
        config.use_weighted_sum = False
        config.agent_individual = False
        config.reverse_state_action = False
        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, trans, device):
        ####################### config basic info
        self.config = config
        self.trans = trans
        self.observation_dim = trans.observation_dim
        self.action_dim = trans.action_dim
        self.device = device
        ####################### config optim
        optimizer_class = {
            'adam': torch.optim.Adam,
            'adamw': torch.optim.AdamW,
            'sgd': torch.optim.SGD,
        }[self.config.optimizer_type]
        self.optimizer = optimizer_class(self.trans.parameters(), lr=self.config.trans_lr)
        ####################### config scheduler
        self.scheduler = {
            'CosineDecay': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=5,  # not sure if setting of scheduler is correct
            ),
            'none': None
        }[self.config.scheduler_type]
        ####################### config other record info
        self._total_steps = 0

    def train(self, batch):
        self.trans.train()
        self._total_steps += 1
        metrics = self._train_pref_step(batch)
        return metrics

    def _train_pref_step(self, batch):
        def loss_fn():
            """
            obs_0 torch.Size([batch_size, seq_len, agent_num, obs_dim])
            act_0 torch.Size([batch_size, seq_len, agent_num, act_dim])
            obs_1 torch.Size([batch_size, seq_len, agent_num, obs_dim])
            act_1 torch.Size([batch_size, seq_len, agent_num, act_dim])
            timestep_0 torch.Size([batch_size, seq_len])
            timestep_1 torch.Size([batch_size, seq_len])
            labels torch.Size([batch_size, 2])
            """
            ####################### get train data from batch
            obs_0 = batch['observations0']
            act_0 = batch['actions0']
            obs_1 = batch['observations1']
            act_1 = batch['actions1']
            timestep_0 = batch['timesteps0']
            timestep_1 = batch['timesteps1']
            labels = batch['labels']
            B, T, N, _ = batch['observations0'].shape
            B, T, N, _ = batch['actions0'].shape
            ####################### copmpute loss
            trans_pred_0, _ = self.trans(obs_0, act_0, timestep_0, training=True, attn_mask=None)
            trans_pred_1, _ = self.trans(obs_1, act_1, timestep_1, training=True, attn_mask=None)
            # print('predict_value', torch.mean(trans_pred_0["value"]))
            # print('predict_value', torch.mean(trans_pred_1["value"]))
            if self.config.use_weighted_sum:
                trans_pred_0 = trans_pred_0["weighted_sum"]
                trans_pred_1 = trans_pred_1["weighted_sum"]
            else:
                trans_pred_0 = trans_pred_0["value"]
                trans_pred_1 = trans_pred_1["value"]
            ####################### add all agents rewards as global reward(or individual reward)
            if self.config.agent_individual:
                trans_pred_0 = trans_pred_0.permute(0, 2, 1, 3).reshape(B * N, T, -1)
                trans_pred_1 = trans_pred_1.permute(0, 2, 1, 3).reshape(B * N, T, -1)
                labels = labels.unsqueeze(1).repeat(1, N, 1).reshape(B * N, -1)
                B = B * N
            else:
                trans_pred_0 = torch.sum(trans_pred_0, dim=-2)
                trans_pred_1 = torch.sum(trans_pred_1, dim=-2)
            if self.config.train_type == "mean":
                sum_pred_0 = torch.mean(trans_pred_0.reshape(B, T), dim=1).reshape(-1, 1)
                sum_pred_1 = torch.mean(trans_pred_1.reshape(B, T), dim=1).reshape(-1, 1)
            elif self.config.train_type == "sum":
                sum_pred_0 = torch.sum(trans_pred_0.reshape(B, T), dim=1).reshape(-1, 1)
                sum_pred_1 = torch.sum(trans_pred_1.reshape(B, T), dim=1).reshape(-1, 1)
            elif self.config.train_type == "last":
                sum_pred_0 = trans_pred_0.reshape(B, T)[:, -1].reshape(-1, 1)
                sum_pred_1 = trans_pred_1.reshape(B, T)[:, -1].reshape(-1, 1)
            logits = torch.cat([sum_pred_0, sum_pred_1], dim=1)
            loss_collection = {}
            trans_loss = cross_ent_loss(logits, labels.detach())
            ####################### copmpute grad and update model
            self.optimizer.zero_grad()
            trans_loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            loss_collection['trans_loss'] = trans_loss.detach().cpu().numpy()
            return loss_collection
        aux_values = loss_fn()
        metrics = dict(
            trans_loss=aux_values['trans_loss'],
        )
        return metrics

    def evaluation(self, batch):
        self.trans.eval()
        metrics = self._eval_pref_step(batch)
        return metrics

    def _eval_pref_step(self, batch):
        def loss_fn():
            ####################### get eval data from batch
            obs_0 = batch['observations0']
            act_0 = batch['actions0']
            obs_1 = batch['observations1']
            act_1 = batch['actions1']
            timestep_0 = batch['timesteps0']
            timestep_1 = batch['timesteps1']
            labels = batch['labels']
            B, T, N, _ = batch['observations0'].shape
            B, T, N, _ = batch['actions0'].shape
            ####################### copmpute loss and grad
            trans_pred_0, _ = self.trans(obs_0, act_0, timestep_0, training=True, attn_mask=None)
            trans_pred_1, _ = self.trans(obs_1, act_1, timestep_1, training=True, attn_mask=None)
            if self.config.use_weighted_sum:
                trans_pred_0 = trans_pred_0["weighted_sum"]
                trans_pred_1 = trans_pred_1["weighted_sum"]
            else:
                trans_pred_0 = trans_pred_0["value"]
                trans_pred_1 = trans_pred_1["value"]
            ####################### add all agents rewards as global reward(or individual reward)
            if self.config.agent_individual:
                # (B * N, T, 1)
                trans_pred_0 = trans_pred_0.permute(0, 2, 1, 3).reshape(B * N, T, -1)
                trans_pred_1 = trans_pred_1.permute(0, 2, 1, 3).reshape(B * N, T, -1)
                labels = labels.unsqueeze(1).repeat(1, N, 1).reshape(B * N, -1)
                B = B * N
            else:
                # (B, T, 1)
                trans_pred_0 = torch.sum(trans_pred_0, dim=-2)
                trans_pred_1 = torch.sum(trans_pred_1, dim=-2)
            if self.config.train_type == "mean":
                sum_pred_0 = torch.mean(trans_pred_0.reshape(B, T), dim=1).reshape(-1, 1)
                sum_pred_1 = torch.mean(trans_pred_1.reshape(B, T), dim=1).reshape(-1, 1)
            elif self.config.train_type == "sum":
                sum_pred_0 = torch.sum(trans_pred_1.reshape(B, T), dim=1).reshape(-1, 1)
                sum_pred_1 = torch.sum(trans_pred_2.reshape(B, T), dim=1).reshape(-1, 1)
            elif self.config.train_type == "last":
                sum_pred_0 = trans_pred_1.reshape(B, T)[:, -1].reshape(-1, 1)
                sum_pred_1 = trans_pred_2.reshape(B, T)[:, -1].reshape(-1, 1)
            logits = torch.cat([sum_pred_0, sum_pred_1], dim=1)
            loss_collection = {}
            trans_loss = cross_ent_loss(logits, labels.detach())
            loss_collection['trans_loss'] = trans_loss.detach().cpu().numpy()
            return loss_collection
        aux_values = loss_fn()
        metrics = dict(
            eval_trans_loss=aux_values['trans_loss'],
        )
        return metrics

    def get_reward(self, batch):
        self.trans.eval()
        return self._get_reward_step(batch)

    def _get_reward_step(self, batch):
        obs = batch['observations']
        act = batch['actions']
        timestep = batch['timestep']
        attn_mask = batch['attn_mask']
        trans_pred, attn_weights = self.trans(
            obs, act, timestep, training=False, attn_mask=attn_mask, reverse=False)

        return trans_pred["value"].squeeze(1)

    ####################### my add method
    def save_model(self, save_path, save_idx):
        torch.save({
            'reward_model': self.trans.state_dict(),
            'seq_len': self.trans.max_episode_steps,
        }, str(save_path) + "reward_model_" + str(save_idx) + ".pt")

    def load_model(self, model_dir):
        model_state_dict = torch.load(model_dir, map_location=torch.device('cpu')) \
            if self.device == torch.device('cpu') else torch.load(model_dir)
        self.trans.load_state_dict(model_state_dict['reward_model'])
        print('--------------- load PrefTransformer -----------------')

    @property
    def total_steps(self):
        return self._total_steps

