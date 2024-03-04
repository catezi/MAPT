import os
import numpy as np
import torch
from collections import defaultdict
from tensorboardX import SummaryWriter
from mat.utils.util import Data_Sampler
from mat.utils.pair_dataloader import load_dataset
from mat.utils.shared_buffer import SharedReplayBuffer
from mat.utils.pref_reward_assistant import PrefRewardAssistant
from mat.algorithms.mat.mat_trainer import MATTrainer as TrainAlgo
from mat.algorithms.reward_model.models.torch_utils import batch_to_torch, index_batch
from mat.algorithms.mat.algorithm.transformer_policy import TransformerPolicy as Policy


def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()


class Runner(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']       

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir
        self.optim_dir = self.all_args.optim_dir

        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        self.share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]

        print("obs_space: ", self.envs.observation_space)
        print("share_obs_space: ", self.envs.share_observation_space)
        print("act_space: ", self.envs.action_space)

        # policy network
        self.policy = Policy(self.all_args,
                             self.envs.observation_space[0],
                             self.share_observation_space,
                             self.envs.action_space[0],
                             self.num_agents,
                             device=self.device)

        if self.model_dir is not None or self.optim_dir is not None:
            self.restore(self.model_dir, self.optim_dir)

        # algorithm
        self.trainer = TrainAlgo(self.all_args, self.policy, self.num_agents, device=self.device)
        
        # buffer
        self.buffer = SharedReplayBuffer(self.all_args,
                                        self.num_agents,
                                        self.envs.observation_space[0],
                                        self.share_observation_space,
                                        self.envs.action_space[0],
                                         self.all_args.env_name)
        config['all_args'].action_type = self.policy.action_type
        self.data_sampler = Data_Sampler(config=config['all_args'])
        self.max_mean_scores = -float("inf")

        ######################### add for preference reward
        self.pref_reward_assistant = PrefRewardAssistant(
            self.all_args, self.envs.observation_space[0],
            self.envs.action_space[0],
            self.num_agents, self.device,
        ) if self.all_args.use_preference_reward else None

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        if self.buffer.available_actions is None:
            next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                         np.concatenate(self.buffer.obs[-1]),
                                                         np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                         np.concatenate(self.buffer.masks[-1]))
        else:
            next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                         np.concatenate(self.buffer.obs[-1]),
                                                         np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                         np.concatenate(self.buffer.masks[-1]),
                                                         np.concatenate(self.buffer.available_actions[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)
    
    def train(self):
        """Train policies with data in buffer. """
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)      
        self.buffer.after_update()
        return train_infos

    def save(self, episode):
        """Save policy's actor and critic networks."""
        self.policy.save(self.save_dir, episode)

    def restore(self, model_dir, optim_dir):
        """Restore policy's networks from a saved model."""
        self.policy.restore(model_dir, optim_dir)
 
    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    ############################# pretrain with pair data
    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)

    def pretrain_pair(self):
        ############################# load pair dataset
        action_type = 'Continous' if (self.all_args.env_name == 'hands' or self.all_args.env_name == 'mujoco') else 'Discrete'
        pref_dataset, pref_eval_dataset, env_info = load_dataset(
            self.all_args.env_name, self.all_args.task, self.all_args.pair_data_dir, action_type
        )
        data_size = pref_dataset["observations0"].shape[0]
        interval = int(data_size / self.all_args.pretrain_batch_size) + 1
        eval_data_size = pref_eval_dataset["observations0"].shape[0]
        eval_interval = int(eval_data_size / self.all_args.pretrain_batch_size) + 1
        min_eval_loss = float('inf')
        #################################### run training pipeline
        for epoch in range(self.all_args.pretrain_epoch):
            metrics = defaultdict(list)
            metrics['epoch'] = epoch
            ####################### train model
            shuffled_idx = np.random.permutation(pref_dataset["observations0"].shape[0])
            for i in range(interval):
                start_pt = i * self.all_args.pretrain_batch_size
                end_pt = min((i + 1) * self.all_args.pretrain_batch_size, pref_dataset["observations0"].shape[0])
                if start_pt >= end_pt:
                    break
                # train
                batch = batch_to_torch(index_batch(pref_dataset, shuffled_idx[start_pt: end_pt]), self.device)
                train_loss = self.trainer.pretrain_pair(batch)
                metrics['train_loss'].append(train_loss)
            ####################### eval model
            if epoch % self.all_args.pretrain_eval_period == 0:
                for j in range(eval_interval):
                    eval_start_pt = j * self.all_args.pretrain_batch_size
                    eval_end_pt = min((j + 1) * self.all_args.pretrain_batch_size, pref_eval_dataset["observations0"].shape[0])
                    batch_eval = batch_to_torch(index_batch(pref_eval_dataset, range(eval_start_pt, eval_end_pt)), self.device)
                    eval_loss = self.trainer.pretrain_pair(batch_eval)
                    metrics['eval_loss'].append(train_loss)
                ####################### save model
                if np.mean(metrics['eval_loss']) < min_eval_loss:
                    min_eval_loss = np.mean(metrics['eval_loss'])
                    self.policy.save(self.save_dir, epoch)
            ####################### log res
            for key, val in metrics.items():
                if isinstance(val, list):
                    metrics[key] = np.mean(val)
            print('########################################')
            for key in metrics:
                print(key, metrics[key])
            self.log_train(metrics, epoch)


