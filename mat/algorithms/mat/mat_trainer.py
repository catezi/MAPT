import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from mat.utils.valuenorm import ValueNorm
from mat.algorithms.utils.util import check
from mat.utils.util import get_gard_norm, huber_loss, mse_loss


class MATTrainer:
    """
    Trainer class for MAT to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self,
                 args,
                 policy,
                 num_agents,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy
        self.num_agents = num_agents

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        self.dec_actor = args.dec_actor
        
        if self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """

        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                    self.clip_param)

        if self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        # if self._use_value_active_masks and not self.dec_actor:
        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch,
                                                                              obs_batch, 
                                                                              rnn_states_batch, 
                                                                              rnn_states_critic_batch, 
                                                                              actions_batch, 
                                                                              masks_batch, 
                                                                              available_actions_batch,
                                                                              active_masks_batch)
        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_loss = (-torch.sum(torch.min(surr1, surr2),
                                      dim=-1,
                                      keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        # critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        loss = policy_loss - dist_entropy * self.entropy_coef + value_loss * self.value_loss_coef

        self.policy.optimizer.zero_grad()
        loss.backward()

        if self._use_max_grad_norm:
            grad_norm = nn.utils.clip_grad_norm_(self.policy.transformer.parameters(), self.max_grad_norm)
        else:
            grad_norm = get_gard_norm(self.policy.transformer.parameters())

        self.policy.optimizer.step()

        return value_loss, grad_norm, policy_loss, dist_entropy, grad_norm, imp_weights

    def train(self, buffer):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        advantages_copy = buffer.advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (buffer.advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {}
        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        for _ in range(self.ppo_epoch):
            data_generator = buffer.feed_forward_generator_transformer(advantages, self.num_mini_batch)
            for sample in data_generator:
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights \
                    = self.ppo_update(sample)
                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()
        num_updates = self.ppo_epoch * self.num_mini_batch
        for k in train_info.keys():
            train_info[k] /= num_updates
 
        return train_info

    def prep_training(self):
        self.policy.train()

    def prep_rollout(self):
        self.policy.eval()

    def pretrain_pair(self, batch):
        """

        """
        ####################### process data from batch
        obs_0, obs_1 = batch['observations0'], batch['observations1']
        act_0, act_1 = batch['actions0'], batch['actions1']
        labels = batch['labels']
        batch_size, seq_len, agent_num = obs_0.shape[0], obs_0.shape[1], obs_0.shape[2]
        obs_0 = obs_0.reshape(batch_size * seq_len, agent_num, -1)
        obs_1 = obs_1.reshape(batch_size * seq_len, agent_num, -1)
        act_0 = act_0.reshape(batch_size * seq_len, agent_num, -1)
        act_1 = act_1.reshape(batch_size * seq_len, agent_num, -1)
        # print('----------------------')
        # print('obs_0', obs_0.shape)
        # print('act_0', act_0.shape)
        # print('labels', labels.shape)
        ####################### use MAT to compute action logprob
        _, action_log_probs_0, _ = self.policy.evaluate_actions(
            cent_obs=torch.zeros((batch_size * seq_len, agent_num, self.policy.share_obs_dim)),
            obs=obs_0, rnn_states_actor=None, rnn_states_critic=None,
            actions=act_0, masks=None, available_actions=None, active_masks=None,
        )
        _, action_log_probs_1, _ = self.policy.evaluate_actions(
            cent_obs=torch.zeros((batch_size * seq_len, agent_num, self.policy.share_obs_dim)),
            obs=obs_1, rnn_states_actor=None, rnn_states_critic=None,
            actions=act_1, masks=None, available_actions=None, active_masks=None,
        )
        ####################### use action logprob to compute loss
        action_log_probs_0 = action_log_probs_0.reshape(batch_size * seq_len, agent_num, -1)
        action_log_probs_0 = action_log_probs_0.reshape(batch_size, seq_len, agent_num, -1)
        action_log_probs_1 = action_log_probs_1.reshape(batch_size * seq_len, agent_num, -1)
        action_log_probs_1 = action_log_probs_1.reshape(batch_size, seq_len, agent_num, -1)
        # compute sum for (action_dim, agents_num, seq_len)
        action_log_probs_0 = action_log_probs_0.mean(dim=-1).mean(dim=-1).mean(dim=-1).unsqueeze(-1)
        action_log_probs_1 = action_log_probs_1.mean(dim=-1).mean(dim=-1).mean(dim=-1).unsqueeze(-1)
        # T1 > T2: labels = 1
        # T2 > T1: labels = -1
        # pair_loss = -labels * action_log_probs_0 + labels * action_log_probs_1
        # pair_loss = -labels * action_log_probs_0 - (1 - labels) * action_log_probs_1
        # pair_loss = pair_loss.mean()
        action_probs = torch.cat([action_log_probs_0.exp(), action_log_probs_1.exp()], dim=-1)
        # print('action_probs', action_probs[:20])
        action_probs = action_probs / 0.1
        pair_loss = F.cross_entropy(input=action_probs, target=labels)
        ####################### update policy para
        self.policy.optimizer.zero_grad()
        pair_loss.backward()
        self.policy.optimizer.step()

        return pair_loss.detach().cpu().numpy()





