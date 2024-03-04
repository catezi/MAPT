import torch
from functools import partial
from ml_collections import ConfigDict
from mat.algorithms.reward_model.models.torch_utils import mse_loss, cross_ent_loss


class MR(object):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.inner_dim = 256
        config.action_embd_dim = 64
        config.rf_lr = 3e-4
        config.optimizer_type = 'adam'
        config.agent_individual = False
        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, rf, device):
        ####################### config basic info
        self.config = config
        self.rf = rf
        self.device = device
        self.observation_dim = rf.observation_dim
        self.action_dim = rf.action_dim
        ####################### config optim
        optimizer_class = {
            'adam': torch.optim.Adam,
            'adamw': torch.optim.AdamW,
            'sgd': torch.optim.SGD,
        }[self.config.optimizer_type]
        self.optimizer = optimizer_class(self.rf.parameters(), lr=self.config.rf_lr)
        ####################### MR not set scheduler
        self._total_steps = 0

    def train(self, batch):
        self.rf.train()
        self._total_steps += 1
        metrics = self._train_pref_step(batch)
        return metrics

    def _train_pref_step(self, batch):
        def loss_fn():
            ####################### get train data from batch
            obs_0 = batch['observations0']
            act_0 = batch['actions0']
            obs_1 = batch['observations1']
            act_1 = batch['actions1']
            labels = batch['labels']
            B, T, N, obs_dim = batch['observations0'].shape
            B, T, N, act_dim = batch['actions0'].shape
            obs_0 = obs_0.reshape(-1, obs_dim)
            obs_1 = obs_1.reshape(-1, obs_dim)
            act_0 = act_0.reshape(-1, act_dim)
            act_1 = act_1.reshape(-1, act_dim)
            ####################### copmpute loss
            rf_pred_0 = self.rf(obs_0, act_0)
            rf_pred_1 = self.rf(obs_1, act_1)
            ####################### add all agents rewards as global reward(or individual reward)
            if self.config.agent_individual:
                # (B * N, T, 1)
                rf_pred_0 = rf_pred_0.reshape(B, T, N, -1).permute(0, 2, 1, 3).reshape(B * N, T, -1)
                rf_pred_1 = rf_pred_1.reshape(B, T, N, -1).permute(0, 2, 1, 3).reshape(B * N, T, -1)
                labels = labels.unsqueeze(1).repeat(1, N, 1).reshape(B * N, -1)
            else:
                # (B, T, 1)
                rf_pred_0 = torch.sum(rf_pred_0.reshape(B, T, N, -1), dim=-2)
                rf_pred_1 = torch.sum(rf_pred_1.reshape(B, T, N, -1), dim=-2)
            sum_pred_0 = torch.mean(rf_pred_0, dim=1).reshape(-1, 1)
            sum_pred_1 = torch.mean(rf_pred_1, dim=1).reshape(-1, 1)
            logits = torch.cat([sum_pred_0, sum_pred_1], dim=1)
            loss_collection = {}
            rf_loss = cross_ent_loss(logits, labels)
            ####################### copmpute grad and update model
            self.optimizer.zero_grad()
            rf_loss.backward()
            self.optimizer.step()
            # self.scheduler.step()
            loss_collection['rf_loss'] = rf_loss.detach().cpu().numpy()
            return loss_collection
        aux_values = loss_fn()
        metrics = dict(
            rf_loss=aux_values['rf_loss'],
        )
        return metrics

    def evaluation(self, batch):
        self.rf.eval()
        metrics = self._eval_pref_step(batch)
        return metrics

    def _eval_pref_step(self, batch):
        def loss_fn():
            ####################### get eval data from batch
            obs_0 = batch['observations0']
            act_0 = batch['actions0']
            obs_1 = batch['observations1']
            act_1 = batch['actions1']
            labels = batch['labels']
            B, T, N, obs_dim = batch['observations0'].shape
            B, T, N, act_dim = batch['actions0'].shape
            obs_0 = obs_0.reshape(-1, obs_dim)
            obs_1 = obs_1.reshape(-1, obs_dim)
            act_0 = act_0.reshape(-1, act_dim)
            act_1 = act_1.reshape(-1, act_dim)
            ####################### copmpute loss
            rf_pred_0 = self.rf(obs_0, act_0)
            rf_pred_1 = self.rf(obs_1, act_1)
            ####################### add all agents rewards as global reward(or individual reward)
            if self.config.agent_individual:
                # (B * N, T, 1)
                rf_pred_0 = rf_pred_0.reshape(B, T, N, -1).permute(0, 2, 1, 3).reshape(B * N, T, -1)
                rf_pred_1 = rf_pred_1.reshape(B, T, N, -1).permute(0, 2, 1, 3).reshape(B * N, T, -1)
                labels = labels.unsqueeze(1).repeat(1, N, 1).reshape(B * N, -1)
            else:
                # (B, T, 1)
                rf_pred_0 = torch.sum(rf_pred_0.reshape(B, T, N, -1), dim=-2)
                rf_pred_1 = torch.sum(rf_pred_1.reshape(B, T, N, -1), dim=-2)
            sum_pred_0 = torch.mean(rf_pred_0, dim=1).reshape(-1, 1)
            sum_pred_1 = torch.mean(rf_pred_1, dim=1).reshape(-1, 1)
            logits = torch.cat([sum_pred_0, sum_pred_1], dim=1)
            loss_collection = {}
            rf_loss = cross_ent_loss(logits, labels)
            loss_collection['rf_loss'] = rf_loss.detach().cpu().numpy()
            return loss_collection
        aux_values = loss_fn()
        metrics = dict(
            eval_rf_loss=aux_values['rf_loss'],
        )
        return metrics

    def get_reward(self, batch):
        self.rf.eval()
        return self._get_reward_step(batch)

    def _get_reward_step(self, batch):
        obs = batch['observations']
        act = batch['actions']
        rf_pred = self.rf(obs, act)
        return rf_pred

    ####################### my add method
    def save_model(self, save_path, save_idx):
        torch.save({
            'reward_model': self.rf.state_dict(),
        }, str(save_path) + "reward_model_" + str(save_idx) + ".pt")

    def load_model(self, model_dir):
        model_state_dict = torch.load(model_dir, map_location=torch.device('cpu')) \
            if self.device == torch.device('cpu') else torch.load(model_dir)
        self.rf.load_state_dict(model_state_dict['reward_model'])
        print('--------------- load MR -----------------')

    @property
    def total_steps(self):
        return self._total_steps

