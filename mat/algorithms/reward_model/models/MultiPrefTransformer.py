import torch
import numpy as np
from ml_collections import ConfigDict
from mat.algorithms.reward_model.models.torch_utils import cross_ent_loss


class MultiPrefTransformer(object):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.trans_lr = 1e-4
        config.optimizer_type = 'adamw'
        config.scheduler_type = 'CosineDecay'
        config.embd_dim = 256
        config.action_embd_dim = 64
        config.n_layer = 1
        config.n_head = 4
        config.atten_dropout = 0.1
        config.resid_dropout = 0.1
        config.pref_attn_embd_dim = 256
        config.medium_process_type = 'cat'
        config.use_weighted_sum = True
        config.reverse_state_action = False
        config.agent_individual = False
        config.use_dropout = False
        config.use_lstm = False
        config.add_obs_action = False
        config.drop_agent_layer = False
        config.use_highway = False
        ############ add config for aboration
        config.encoder_mlp = False
        config.decoder_mlp = False
        ############ add config for MPTD aboration
        config.agent_layer_mlp = False
        config.time_layer_mlp = False
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
            'StepLR': torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=100, gamma=0.98,
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
            trans_pred_0 = self.trans(obs_0, act_0, timestep_0, attn_mask=None)
            trans_pred_1 = self.trans(obs_1, act_1, timestep_1, attn_mask=None)
            ####################### add all agents rewards as global reward(or individual reward)
            if self.config.agent_individual:
                trans_pred_0 = trans_pred_0.permute(0, 2, 1, 3).reshape(B * N, -1)
                trans_pred_1 = trans_pred_1.permute(0, 2, 1, 3).reshape(B * N, -1)
                labels = labels.reshape(B * N, -1)
            else:
                trans_pred_0 = torch.sum(trans_pred_0, dim=-2).reshape(B, -1)
                trans_pred_1 = torch.sum(trans_pred_1, dim=-2).reshape(B, -1)
            ####################### add all tiemsteps value to evaluate a sequencec
            trans_pred_0 = torch.sum(trans_pred_0.reshape(B, T), dim=1).reshape(-1, 1)
            trans_pred_1 = torch.sum(trans_pred_1.reshape(B, T), dim=1).reshape(-1, 1)
            logits = torch.cat([trans_pred_0, trans_pred_1], dim=1)
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
            trans_pred_0 = self.trans(obs_0, act_0, timestep_0, attn_mask=None)
            trans_pred_1 = self.trans(obs_1, act_1, timestep_1, attn_mask=None)
            ####################### add all agents rewards as global reward(or individual reward)
            if self.config.agent_individual:
                trans_pred_0 = trans_pred_0.permute(0, 2, 1, 3).reshape(B * N, -1)
                trans_pred_1 = trans_pred_1.permute(0, 2, 1, 3).reshape(B * N, -1)
                labels = labels.reshape(B * N, -1)
            else:
                trans_pred_0 = torch.sum(trans_pred_0, dim=-2).reshape(B, -1)
                trans_pred_1 = torch.sum(trans_pred_1, dim=-2).reshape(B, -1)
            ####################### add all tiemsteps value to evaluate a sequencec
            trans_pred_0 = torch.sum(trans_pred_0.reshape(B, T), dim=1).reshape(-1, 1)
            trans_pred_1 = torch.sum(trans_pred_1.reshape(B, T), dim=1).reshape(-1, 1)
            logits = torch.cat([trans_pred_0, trans_pred_1], dim=1)
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
        trans_pred = self.trans(obs, act, timestep, attn_mask=attn_mask)

        return trans_pred.squeeze(1)

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
