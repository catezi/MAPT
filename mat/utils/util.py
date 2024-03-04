from torch.nn import functional as F
import numpy as np
import functools
import math
import torch
import pickle
# import h5py
import os


def check(input):
    if type(input) == np.ndarray:
        return torch.from_numpy(input)


def get_gard_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)


def mse_loss(e):
    return e**2/2


def get_shape_from_obs_space(obs_space):
    if obs_space.__class__.__name__ == 'Box':
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == 'list':
        obs_shape = obs_space
    else:
        raise NotImplementedError
    return obs_shape


def get_shape_from_act_space(act_space):
    if act_space.__class__.__name__ == 'Discrete':
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    else:  # agar
        act_shape = act_space[0].shape[0] + 1
    return act_shape


def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N)/H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0]*0 for _ in range(N, H*W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H*h, W*w, c)
    return img_Hh_Ww_c


class Data_Sampler:
    def __init__(self, config):
        self.config = config
        self.all_step_data = [{
            'state': [],
            'share_state': [],
            'action': [],
            'reward': [],
            'next_state': [],
            'next_share_state': [],
            'done': [],
            'episode_reward': 0,
            # add extra data for icq
            'action_onehot': [],
            'avail_action': [],
        } for _ in range(self.config.n_eval_rollout_threads)]
        self.all_episode_data = []
        self.total_episodes = 0
        self.total_steps = 0
        self.total_need_data = self.config.total_sample_steps
        self.total_sample_data = int(self.config.total_sample_steps / self.config.choose_rate)
        # add extra data for icq
        self.max_ep_len = 0

    def add_step(self, state, share_state, action, reward,
                 next_state, next_share_state, done):
        # print('------------------------')
        # print('state', state.shape)
        # print('share_state', share_state.shape)
        # print('action', action.shape)
        # print('reward', reward.shape)
        # print('next_state', next_state.shape)
        # print('share_next_state', next_share_state.shape)
        # print('done', done.shape)
        for eval_i in range(self.config.n_eval_rollout_threads):
            # add step data for every type
            self.all_step_data[eval_i]['state'].append(state[eval_i])
            self.all_step_data[eval_i]['share_state'].append(share_state[eval_i])
            self.all_step_data[eval_i]['action'].append(action[eval_i])
            self.all_step_data[eval_i]['reward'].append(reward[eval_i])
            self.all_step_data[eval_i]['next_state'].append(next_state[eval_i])
            self.all_step_data[eval_i]['next_share_state'].append(next_share_state[eval_i])
            self.all_step_data[eval_i]['done'].append(done[eval_i])
            # finish an episode and add to episode data
            if done[eval_i]:
                self.total_steps += len(self.all_step_data[eval_i]['state'])
                self.total_episodes += 1
                self.all_step_data[eval_i]['episode_reward'] = np.sum(np.mean(np.stack(
                    self.all_step_data[eval_i]['reward'], axis=0
                ), axis=1).flatten())
                # print('episode_len', len(self.all_step_data[eval_i]['state']))
                # print('episode_reward', self.all_step_data[eval_i]['episode_reward'])
                self.all_episode_data.append(self.all_step_data[eval_i])
                self.all_step_data[eval_i] = {
                    'state': [],
                    'share_state': [],
                    'action': [],
                    'reward': [],
                    'next_state': [],
                    'next_share_state': [],
                    'done': [],
                    'episode_reward': 0,
                }
        if self.total_steps >= self.total_sample_data:
            self.save_data()
            return True, self.total_steps
        else:
            return False, self.total_steps

    def save_data(self, ):
        print('total steps num', self.total_steps)
        print('total episodes num', len(self.all_episode_data))
        self.all_episode_data = sorted(self.all_episode_data, key=functools.cmp_to_key(lambda ep1, ep2: 1 if ep1['episode_reward'] < ep2['episode_reward'] else -1))
        save_episode_data = []
        save_episode_rewards = []
        save_total_steps_num = 0
        save_total_episodes_num = 0
        for episode_data in self.all_episode_data:
            save_episode_data.append(episode_data)
            save_episode_rewards.append(episode_data['episode_reward'])
            save_total_steps_num += len(episode_data['state'])
            save_total_episodes_num += 1
            if save_total_steps_num >= self.total_need_data:
                break
        print('save total steps num', save_total_steps_num)
        print('save total episodes num', save_total_episodes_num)
        print('save mean episodes reward', np.mean(save_episode_rewards))

        # save episode data to dir
        if not os.path.isdir(self.config.sample_data_dir):
            os.makedirs(self.config.sample_data_dir)
        with open(self.config.sample_data_dir + '/' + str(self.config.env_name + '.pkl'), 'wb') as f:
            pickle.dump(save_episode_data, f)

    def add_step_icq(self, state, share_state, action, reward,
                     next_state, next_share_state, done, avail_actions):
        """
        obs_h torch.Size([100, 151, 3, 36])
        reward_h torch.Size([100, 151, 1])
        actions_h torch.Size([100, 151, 3, 1])
        actions_onehot_h torch.Size([100, 151, 3, 9])
        avail_actions_h torch.Size([100, 151, 3, 9])
        filled_h torch.Size([100, 151, 1])
        state_h torch.Size([100, 151, 27])
        terminated_h torch.Size([100, 151, 1])

        state (40, 249, 2, 218)
        share_state (40, 249, 417)
        action (40, 249, 2, 26)
        reward (40, 249, 1)
        action_onehot (40, 249, 2, 26)
        avail_action (40, 249, 2, 26)
        done (40, 249, 1)
        fill (40, 249, 1)
        """
        all_done = np.all(done, axis=1)
        reward = np.mean(reward, axis=1)
        for eval_i in range(self.config.n_eval_rollout_threads):
            # add step data for every type
            self.all_step_data[eval_i]['state'].append(state[eval_i])
            self.all_step_data[eval_i]['share_state'].append(share_state[eval_i][0])
            self.all_step_data[eval_i]['action'].append(action[eval_i])
            self.all_step_data[eval_i]['reward'].append(reward[eval_i])
            self.all_step_data[eval_i]['next_state'].append(next_state[eval_i])
            self.all_step_data[eval_i]['next_share_state'].append(next_share_state[eval_i][0])
            self.all_step_data[eval_i]['done'].append(np.expand_dims(all_done[eval_i], axis=-1))
            # add extra data for icq
            self.all_step_data[eval_i]['action_onehot'].append(
                np.array(F.one_hot(torch.from_numpy(action[eval_i]).squeeze(-1), num_classes=avail_actions.shape[-1]))
                if self.config.action_type == 'Discrete' else np.ones_like(action[eval_i])
            )
            self.all_step_data[eval_i]['avail_action'].append(avail_actions[eval_i])
            # finish an episode and add to episode data
            if all_done[eval_i]:
                self.total_steps += len(self.all_step_data[eval_i]['state'])
                self.total_episodes += 1
                self.all_step_data[eval_i]['episode_reward'] = np.sum(np.mean(np.stack(
                    self.all_step_data[eval_i]['reward'], axis=0
                ), axis=1).flatten())
                # print('episode_len', len(self.all_step_data[eval_i]['state']))
                # print('episode_reward', self.all_step_data[eval_i]['episode_reward'])
                # print('------------------------')
                # print('state', np.array(self.all_step_data[eval_i]['state']).shape)
                # print('share_state', np.array(self.all_step_data[eval_i]['share_state']).shape)
                # print('action', np.array(self.all_step_data[eval_i]['action']).shape)
                # print('reward', np.array(self.all_step_data[eval_i]['reward']).shape)
                # print('action_onehot', np.array(self.all_step_data[eval_i]['action_onehot']).shape)
                # print('avail_action', np.array(self.all_step_data[eval_i]['avail_action']).shape)
                # print('done', np.array(self.all_step_data[eval_i]['done']).shape)
                self.all_episode_data.append(self.all_step_data[eval_i])
                self.max_ep_len = max(self.max_ep_len, len(self.all_step_data[eval_i]['state']))
                self.all_step_data[eval_i] = {
                    'state': [],
                    'share_state': [],
                    'action': [],
                    'reward': [],
                    'next_state': [],
                    'next_share_state': [],
                    'done': [],
                    'episode_reward': 0,
                    # add extra data for icq
                    'action_onehot': [],
                    'avail_action': [],
                }
        if self.total_steps >= self.total_sample_data:
            self.save_data_icq()
            return True, self.total_steps
        else:
            return False, self.total_steps

    def save_data_icq(self):
        print('total steps num', self.total_steps)
        print('total episodes num', len(self.all_episode_data))
        save_episode_data = {
            'state': [],
            'share_state': [],
            'action': [],
            'reward': [],
            'next_state': [],
            'next_share_state': [],
            'done': [],
            'action_onehot': [],
            'avail_action': [],
            'fill': [],
        }
        for seq_data in self.all_episode_data:
            """
            filled_h torch.Size([100, 151, 1])
            """
            # transfer data type
            seq_data['state'] = np.array(seq_data['state'])
            seq_data['share_state'] = np.array(seq_data['share_state'])
            seq_data['next_state'] = np.array(seq_data['next_state'])
            seq_data['next_share_state'] = np.array(seq_data['next_share_state'])
            seq_data['action'] = np.array(seq_data['action'])
            seq_data['reward'] = np.array(seq_data['reward'])
            seq_data['action_onehot'] = np.array(seq_data['action_onehot'])
            seq_data['avail_action'] = np.array(seq_data['avail_action'])
            seq_data['done'] = np.array(seq_data['done'])

            now_len = seq_data['state'].shape[0]
            save_episode_data['state'].append(np.expand_dims(np.concatenate([
                seq_data['state'], np.zeros((self.max_ep_len - now_len, *seq_data['state'].shape[1:]))
            ], axis=0), axis=0))
            save_episode_data['share_state'].append(np.expand_dims(np.concatenate([
                seq_data['share_state'], np.zeros((self.max_ep_len - now_len, *seq_data['share_state'].shape[1:]))
            ], axis=0), axis=0))
            save_episode_data['next_state'].append(np.expand_dims(np.concatenate([
                seq_data['next_state'], np.zeros((self.max_ep_len - now_len, *seq_data['next_state'].shape[1:]))
            ], axis=0), axis=0))
            save_episode_data['next_share_state'].append(np.expand_dims(np.concatenate([
                seq_data['next_share_state'], np.zeros((self.max_ep_len - now_len, *seq_data['next_share_state'].shape[1:]))
            ], axis=0), axis=0))
            save_episode_data['action'].append(np.expand_dims(np.concatenate([
                seq_data['action'], np.zeros((self.max_ep_len - now_len, *seq_data['action'].shape[1:]))
            ], axis=0), axis=0))
            save_episode_data['reward'].append(np.expand_dims(np.concatenate([
                seq_data['reward'], np.zeros((self.max_ep_len - now_len, *seq_data['reward'].shape[1:]))
            ], axis=0), axis=0))
            save_episode_data['action_onehot'].append(np.expand_dims(np.concatenate([
                seq_data['action_onehot'], np.zeros((self.max_ep_len - now_len, *seq_data['action_onehot'].shape[1:]))
            ], axis=0), axis=0))
            save_episode_data['avail_action'].append(np.expand_dims(np.concatenate([
                seq_data['avail_action'], np.zeros((self.max_ep_len - now_len, *seq_data['avail_action'].shape[1:]))
            ], axis=0), axis=0))
            save_episode_data['done'].append(np.expand_dims(np.concatenate([
                seq_data['done'], np.zeros((self.max_ep_len - now_len, *seq_data['done'].shape[1:]))
            ], axis=0), axis=0))
            save_episode_data['fill'].append(np.expand_dims(np.concatenate([
                np.ones((now_len, 1)), np.zeros((self.max_ep_len - now_len, 1))
            ], axis=0), axis=0))

        # cat all data
        save_episode_data['state'] = np.concatenate(save_episode_data['state'], axis=0)
        save_episode_data['share_state'] = np.concatenate(save_episode_data['share_state'], axis=0)
        save_episode_data['next_state'] = np.concatenate(save_episode_data['next_state'], axis=0)
        save_episode_data['next_share_state'] = np.concatenate(save_episode_data['next_share_state'], axis=0)
        save_episode_data['action'] = np.concatenate(save_episode_data['action'], axis=0)
        save_episode_data['reward'] = np.concatenate(save_episode_data['reward'], axis=0)
        save_episode_data['action_onehot'] = np.concatenate(save_episode_data['action_onehot'], axis=0)
        save_episode_data['avail_action'] = np.concatenate(save_episode_data['avail_action'], axis=0)
        save_episode_data['done'] = np.concatenate(save_episode_data['done'], axis=0)
        save_episode_data['fill'] = np.concatenate(save_episode_data['fill'], axis=0)

        print('state', save_episode_data['state'].shape)
        print('share_state', save_episode_data['share_state'].shape)
        print('action', save_episode_data['action'].shape)
        print('reward', save_episode_data['reward'].shape)
        print('action_onehot', save_episode_data['action_onehot'].shape)
        print('avail_action', save_episode_data['avail_action'].shape)
        print('done', save_episode_data['done'].shape)
        print('fill', save_episode_data['fill'].shape)

        # save episode data to dir
        if not os.path.isdir(self.config.sample_data_dir):
            os.makedirs(self.config.sample_data_dir)
        with open(self.config.sample_data_dir + '/' + str(self.config.env_name + '.pkl'), 'wb') as f:
            pickle.dump(save_episode_data, f)


class PairDataSampler:
    def __init__(self):
        pass


all_model_path = {
    'ShadowHandDoorOpenInward': [
        "/home/LAB/qiuyue/Multi-Agent-Transformer-main/mat/scripts/results/hands/"
        "ShadowHandDoorOpenInward/mat/single/run1/models/transformer_0.pt",

        "/home/LAB/qiuyue/Multi-Agent-Transformer-main/mat/scripts/results/hands/"
        "ShadowHandDoorOpenInward/mat/single/run1/models/transformer_100.pt",

        "/home/LAB/qiuyue/Multi-Agent-Transformer-main/mat/scripts/results/hands/"
        "ShadowHandDoorOpenInward/mat/single/run3/models/transformer_200.pt",

        "/home/LAB/qiuyue/Multi-Agent-Transformer-main/mat/scripts/results/hands/"
        "ShadowHandDoorOpenInward/mat/single/run6/models/transformer_600.pt",
    ],
    'ShadowHandDoorCloseOutward': [
        "/home/LAB/qiuyue/Multi-Agent-Transformer-main/mat/scripts/results/hands/"
        "ShadowHandDoorCloseOutward/mat/single/run1/models/transformer_0.pt",

        "/home/LAB/qiuyue/Multi-Agent-Transformer-main/mat/scripts/results/hands/"
        "ShadowHandDoorCloseOutward/mat/single/run1/models/transformer_100.pt",

        "/home/LAB/qiuyue/Multi-Agent-Transformer-main/mat/scripts/results/hands/"
        "ShadowHandDoorCloseOutward/mat/single/run2/models/transformer_700.pt",

        "/home/LAB/qiuyue/Multi-Agent-Transformer-main/mat/scripts/results/hands/"
        "ShadowHandDoorCloseOutward/mat/single/run3/models/transformer_1800.pt",
    ],
    'ShadowHandCatchOver2Underarm': [
        "/home/LAB/qiuyue/Multi-Agent-Transformer-main/mat/scripts/results/hands/"
        "ShadowHandCatchOver2Underarm/mat/single/run1/models/transformer_100.pt",

        "/home/LAB/qiuyue/Multi-Agent-Transformer-main/mat/scripts/results/hands/"
        "ShadowHandCatchOver2Underarm/mat/single/run2/models/transformer_1000.pt",

        "/home/LAB/qiuyue/Multi-Agent-Transformer-main/mat/scripts/results/hands/"
        "ShadowHandCatchOver2Underarm/mat/single/run3/models/transformer_200.pt",

        "/home/LAB/qiuyue/Multi-Agent-Transformer-main/mat/scripts/results/hands/"
        "ShadowHandCatchOver2Underarm/mat/single/run3/models/transformer_8332.pt",
    ]
}

