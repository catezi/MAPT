import os
import sys
import gym
import torch
import pickle
import absl.app
import absl.flags
import numpy as np
import transformers
from collections import defaultdict
from tensorboardX import SummaryWriter
sys.path.append("../../")
from mat.algorithms.reward_model.models.MR import MR
from mat.algorithms.reward_model.models.NMR import NMR
from mat.algorithms.reward_model.models.lstm import LSTMRewardModel
from mat.algorithms.reward_model.utils.dataloader import load_dataset
from mat.algorithms.reward_model.models.PrefTransformer import PrefTransformer
from mat.algorithms.reward_model.models.trajectory_gpt2 import TransRewardModel
from mat.algorithms.reward_model.models.q_function import FullyConnectedQFunction
from mat.algorithms.reward_model.models.torch_utils import batch_to_torch, index_batch
from mat.algorithms.reward_model.models.MultiPrefTransformer import MultiPrefTransformer
from mat.algorithms.reward_model.models.encoder_decoder_divide import MultiTransRewardDivideModel
from mat.algorithms.reward_model.utils.utils import Timer, define_flags_with_default, set_random_seed, \
    get_user_flags, prefix_metrics, WandBLogger, save_pickle


FLAGS_DEF = define_flags_with_default(
    # env='hands',
    # task='ShadowHandDoorOpenInward',
    env='smac',
    task='3m',
    model_type='MultiPrefTransformerDivide',
    seed=42,
    save_model=True,
    ################ hyper parameters
    batch_size=256,
    orthogonal_init=False,
    activations='relu',
    activation_final='none',
    n_epochs=2000,
    eval_period=5,
    comment='train',
    max_traj_length=100,
    multi_transformer=MultiPrefTransformer.get_default_config(),
    transformer=PrefTransformer.get_default_config(),
    reward=MR.get_default_config(),
    lstm=NMR.get_default_config(),
    logging=WandBLogger.get_default_config(),
    ################ dir config
    device='cpu',
    dataset_path='D:\Research\preference_refer\MAPT-master\data\smac\\3m_replaybuffer_1w_extre\prefrenece_pair_data.pkl',
    model_dir="",
)


def main(_):
    #################################### set basic config
    FLAGS = absl.flags.FLAGS
    #################### set device
    FLAGS.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #################### set logger dir
    save_dir = FLAGS.logging.output_dir + '/' + FLAGS.env + '/' + FLAGS.task
    save_dir += '/' + str(FLAGS.model_type) + '/'
    save_dir += f"{FLAGS.comment}" + "/"
    FLAGS.logging.group = f"{FLAGS.env}_{FLAGS.model_type}"
    assert FLAGS.comment, "You must leave your comment for logging experiment."
    FLAGS.logging.group += f"_{FLAGS.comment}"
    FLAGS.logging.experiment_id = FLAGS.logging.group + f"_s{FLAGS.seed}"
    FLAGS.logging.log_dir = save_dir + '/logs/'
    FLAGS.logging.model_dir = save_dir + '/models/'
    if not os.path.exists(FLAGS.logging.log_dir):
        os.makedirs(FLAGS.logging.log_dir)
    if not os.path.exists(FLAGS.logging.model_dir):
        os.makedirs(FLAGS.logging.model_dir)
    writter = SummaryWriter(FLAGS.logging.log_dir)
    #################### set random seed
    set_random_seed(FLAGS.seed)
    #################################### load dataset
    action_type = 'Continous' if (FLAGS.env == 'hands' or FLAGS.env == 'mujoco') else 'Discrete'
    pref_dataset, pref_eval_dataset, env_info = load_dataset(
        FLAGS.env, FLAGS.task, FLAGS.dataset_path, action_type
    )
    data_size = pref_dataset["observations0"].shape[0]
    interval = int(data_size / FLAGS.batch_size) + 1
    eval_data_size = pref_eval_dataset["observations0"].shape[0]
    eval_interval = int(eval_data_size / FLAGS.batch_size) + 1
    print('----------------------  finish load data')
    #################################### set env info
    observation_dim, action_dim = env_info['observation_dim'], env_info['action_dim']
    FLAGS.max_traj_length = env_info['max_len']
    n_agent = env_info['n_agent']
    #################################### config reward model
    if FLAGS.model_type == "MultiPrefTransformerDivide":
        config = transformers.GPT2Config(**FLAGS.multi_transformer)
        # config multi-transformer reward model
        trans = MultiTransRewardDivideModel(
            config=config, observation_dim=observation_dim, action_dim=action_dim, n_agent=n_agent,
            action_type=action_type, max_episode_steps=FLAGS.max_traj_length, device=FLAGS.device,
        )
        # config model wrapper for train and eval
        reward_model = MultiPrefTransformer(config, trans, FLAGS.device)
    elif FLAGS.model_type == "PrefTransformer":
        config = transformers.GPT2Config(**FLAGS.transformer)
        config.warmup_steps = int(FLAGS.n_epochs * 0.1 * interval)
        config.total_steps = FLAGS.n_epochs * interval
        # config transformer reward model
        trans = TransRewardModel(
            config=config, observation_dim=observation_dim, action_dim=action_dim, action_type=action_type,
            activation=FLAGS.activations, activation_final=FLAGS.activation_final,
            max_episode_steps=FLAGS.max_traj_length, device=FLAGS.device,
        )
        # config model wrapper for train and eval
        reward_model = PrefTransformer(config, trans, FLAGS.device)
    elif FLAGS.model_type == "MR":
        rf = FullyConnectedQFunction(
            observation_dim=observation_dim, action_dim=action_dim, action_type=action_type,
            inner_dim=FLAGS.reward.inner_dim, action_embd_dim=FLAGS.reward.action_embd_dim,
            orthogonal_init=FLAGS.orthogonal_init, activations=FLAGS.activations,
            activation_final=FLAGS.activation_final, device=FLAGS.device,
        )
        reward_model = MR(FLAGS.reward, rf, FLAGS.device)
    elif FLAGS.model_type == "NMR":
        config = transformers.GPT2Config(**FLAGS.lstm)
        config.warmup_steps = int(FLAGS.n_epochs * 0.1 * interval)
        config.total_steps = FLAGS.n_epochs * interval
        lstm = LSTMRewardModel(
            config=config, observation_dim=observation_dim, action_dim=action_dim, action_type=action_type,
            activation=FLAGS.activations, activation_final=FLAGS.activation_final,
            max_episode_steps=FLAGS.max_traj_length, device=FLAGS.device,
        )
        reward_model = NMR(config, lstm, FLAGS.device)
    else:
        raise NotImplementedError()

    if FLAGS.model_type == "MultiPrefTransformerDivide":
        eval_loss = "reward/eval_trans_loss"
    elif FLAGS.model_type == "PrefTransformer":
        eval_loss = "reward/eval_trans_loss"
    elif FLAGS.model_type == "MR":
        eval_loss = "reward/eval_rf_loss"
    elif FLAGS.model_type == "NMR":
        eval_loss = "reward/eval_lstm_loss"
    min_eval_loss = float('inf')
    #################################### run training pipeline
    for epoch in range(FLAGS.n_epochs):
        metrics = defaultdict(list)
        metrics['epoch'] = epoch
        ####################### train model
        shuffled_idx = np.random.permutation(pref_dataset["observations0"].shape[0])
        for i in range(interval):
            start_pt = i * FLAGS.batch_size
            end_pt = min((i + 1) * FLAGS.batch_size, pref_dataset["observations0"].shape[0])
            if start_pt >= end_pt:
                break
            with Timer() as train_timer:
                # train
                batch = batch_to_torch(index_batch(pref_dataset, shuffled_idx[start_pt: end_pt]), FLAGS.device)
                for key, val in prefix_metrics(reward_model.train(batch), 'reward').items():
                    metrics[key].append(val)
        ####################### eval model
        if epoch % FLAGS.eval_period == 0:
            for j in range(eval_interval):
                eval_start_pt = j * FLAGS.batch_size
                eval_end_pt = min((j + 1) * FLAGS.batch_size, pref_eval_dataset["observations0"].shape[0])
                batch_eval = batch_to_torch(index_batch(pref_eval_dataset, range(eval_start_pt, eval_end_pt)), FLAGS.device)
                for key, val in prefix_metrics(reward_model.evaluation(batch_eval), 'reward').items():
                    metrics[key].append(val)
        ####################### save model
            if FLAGS.save_model and np.mean(metrics[eval_loss]) < min_eval_loss:
                min_eval_loss = np.mean(metrics[eval_loss])
                reward_model.save_model(FLAGS.logging.model_dir, epoch)
        ####################### log res
        print('---------------------- {epoch}', epoch)
        for key, val in metrics.items():
            if isinstance(val, list):
                metrics[key] = np.mean(val)
                print(key, np.mean(val))
        # print('##############################')
        # for key in metrics:
        #     print(key, metrics[key])
        output_res(writter, metrics, epoch)


def output_res(writter, train_infos, step):
    for k, v in train_infos.items():
        writter.add_scalars(k, {k: v}, step)


if __name__ == '__main__':
    absl.app.run(main)



