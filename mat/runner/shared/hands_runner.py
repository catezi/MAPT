import os
import time
import copy
import torch
import pickle
import numpy as np
from functools import reduce
from mat.runner.shared.base_runner import Runner
from mat.utils.pref_reward_assistant import RewardCollector


def _t2n(x):
    return x.detach().cpu().numpy()


class HandsRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""
    def __init__(self, config):
        super(HandsRunner, self).__init__(config)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        train_episode_rewards = [0 for _ in range(self.n_rollout_threads)]
        pref_episode_rewards = [0 for _ in range(self.n_rollout_threads)] \
            if self.all_args.use_preference_reward else None
        done_episodes_rewards = []
        done_pref_episodes_rewards = []
        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)

                # Obser reward and next obs
                obs, share_obs, rewards, dones, infos, available_actions = \
                    self.envs.step(torch.tensor(actions.transpose((1, 0, 2))))

                obs = _t2n(obs)
                share_obs = _t2n(share_obs)
                rewards = _t2n(rewards)
                dones = _t2n(dones)
                dones_env = np.all(dones, axis=1)
                reward_env = np.mean(rewards, axis=1).flatten()
                train_episode_rewards += reward_env
                for t in range(self.n_rollout_threads):
                    if dones_env[t]:
                        done_episodes_rewards.append(train_episode_rewards[t])
                        train_episode_rewards[t] = 0

                if self.all_args.use_preference_reward:
                    ######################### preference: prefreence reward model
                    pref_reward = self.pref_reward_assistant.insert(obs, actions, dones)
                    self.pref_reward_assistant.clear_done()
                    ######################### preference: save preference reward to log
                    pref_episode_rewards += np.mean(pref_reward, axis=1).flatten()
                    for t in range(self.n_rollout_threads):
                        if dones_env[t]:
                            done_pref_episodes_rewards.append(pref_episode_rewards[t])
                            pref_episode_rewards[t] = 0

                ######################### preference: replace reward to buffer with preference reward
                if self.all_args.use_preference_reward:
                    data = obs, share_obs, pref_reward, dones, infos, available_actions, \
                           values, actions, action_log_probs, \
                           rnn_states, rnn_states_critic
                else:
                    data = obs, share_obs, rewards, dones, infos, available_actions, \
                           values, actions, action_log_probs, \
                           rnn_states, rnn_states_critic

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            # # save model
            # if (episode % self.save_interval == 0 or episode == episodes - 1):
            #     self.save(total_num_steps)

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Task {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.task,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                self.log_train(train_infos, total_num_steps)

                if len(done_episodes_rewards) > 0:
                    aver_episode_rewards = np.mean(done_episodes_rewards)
                    print("some episodes done, average rewards: ", aver_episode_rewards)
                    self.writter.add_scalars("train_episode_rewards", {"aver_rewards": aver_episode_rewards}, total_num_steps)
                    # save model
                    if aver_episode_rewards > self.max_mean_scores:
                        self.max_mean_scores = aver_episode_rewards
                        self.save(total_num_steps)
                    done_episodes_rewards = []

                if len(done_pref_episodes_rewards) > 0:
                    aver_pref_episode_rewards = np.mean(done_pref_episodes_rewards)
                    print("some episodes done, average pref rewards: ", aver_pref_episode_rewards)
                    self.writter.add_scalars("train_pref_episode_rewards", {"aver_rewards": aver_pref_episode_rewards}, total_num_steps)
                    done_pref_episodes_rewards = []

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs, share_obs, _ = self.envs.reset()
        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs
        self.buffer.share_obs[0] = _t2n(share_obs).copy()
        self.buffer.obs[0] = _t2n(obs).copy()
        ######################### add for preference reward
        if self.all_args.use_preference_reward:
            self.pref_reward_assistant.his_next_obs[:, self.pref_reward_assistant.preference_traj_length - 1] = _t2n(obs).copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_state, rnn_state_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                            np.concatenate(self.buffer.obs[step]),
                                            np.concatenate(self.buffer.rnn_states[step]),
                                            np.concatenate(self.buffer.rnn_states_critic[step]),
                                            np.concatenate(self.buffer.masks[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        # bad_masks = np.array([[[0.0] if info[agent_id]['bad_transition'] else [1.0] for agent_id in range(self.num_agents)] for info in infos])

        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic,
                           actions, action_log_probs, values, rewards, masks, None, active_masks,
                           None)

    def log_train(self, train_infos, total_num_steps):
        train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
        print("average_step_rewards is {}.".format(train_infos["average_step_rewards"]))
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    @torch.no_grad()
    def sample_data(self):
        eval_episode = 0
        eval_episode_rewards = []
        one_episode_rewards = []
        eval_episode_lengths = []
        one_episode_lengths = []
        for eval_i in range(self.n_eval_rollout_threads):
            one_episode_rewards.append([])
            one_episode_lengths.append(0)

        eval_obs, eval_share_obs, _ = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            self.trainer.prep_rollout()
            eval_actions, eval_rnn_states = \
                self.trainer.policy.act(np.concatenate(eval_share_obs.cpu().numpy()),
                                        np.concatenate(eval_obs.cpu().numpy()),
                                        np.concatenate(eval_rnn_states),
                                        np.concatenate(eval_masks),
                                        deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            # save now state and get next state
            eval_now_obs, now_eval_share_obs = eval_obs, eval_share_obs
            # Obser reward and next obs

            eval_obs, eval_share_obs, eval_rewards, eval_dones, _, _ = self.eval_envs.step(torch.tensor(eval_actions.transpose(1, 0, 2)))
            eval_rewards = eval_rewards.cpu().numpy()
            eval_dones = eval_dones.cpu().numpy()

            # print('-----------------------------')
            # print('eval_obs', eval_obs.shape)
            # print('eval_share_obs', eval_share_obs.shape)
            # print('eval_rewards', eval_rewards.shape)
            # print('eval_rewards', eval_rewards[1])
            # all_episode_rewards += np.mean(eval_rewards, axis=1).flatten()
            # print('eval_dones', eval_dones.shape)
            # print('n_eval_rollout_threads', self.all_args.n_eval_rollout_threads)
            eval_dones_env = np.all(eval_dones, axis=1)
            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            if self.all_args.sample_icq_data:
                enough_data, total_steps_num = self.data_sampler.add_step_icq(
                    state=eval_now_obs.cpu().numpy(),
                    share_state=now_eval_share_obs.cpu().numpy(),
                    action=eval_actions,
                    reward=eval_rewards,
                    next_state=eval_obs.cpu().numpy(),
                    next_share_state=eval_share_obs.cpu().numpy(),
                    # done=np.all(eval_dones, axis=1),
                    done=eval_dones,
                    avail_actions=np.ones_like(eval_actions),
                )
            else:
                # add data to data sampler
                enough_data, total_steps_num = self.data_sampler.add_step(
                    state=eval_now_obs.cpu().numpy(),
                    share_state=now_eval_share_obs.cpu().numpy(),
                    action=eval_actions,
                    reward=eval_rewards,
                    next_state=eval_obs.cpu().numpy(),
                    next_share_state=eval_share_obs.cpu().numpy(),
                    done=np.all(eval_dones, axis=1),
                )

            if eval_episode % 10 == 0:
                print('total_steps_num', total_steps_num)

            for eval_i in range(self.n_eval_rollout_threads):
                one_episode_rewards[eval_i].append(np.mean(eval_rewards[eval_i]))
                one_episode_lengths[eval_i] += 1

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(np.sum(one_episode_rewards[eval_i]))
                    one_episode_rewards[eval_i] = []
                    eval_episode_lengths.append(np.sum(one_episode_lengths[eval_i]))
                    one_episode_lengths[eval_i] = []

            # if eval_steps >= self.all_args.total_sample_steps:
            if enough_data:
                # eval_episode_rewards = torch.cat(eval_episode_rewards, dim=-1)
                # eval_episode_rewards = torch.cat(eval_episode_rewards)
                eval_env_infos = {'eval_average_episode_rewards': np.mean(eval_episode_rewards),
                                  'eval_max_episode_rewards': np.max(eval_episode_rewards)}
                print(eval_env_infos)
                # self.log_env(eval_env_infos, total_num_steps)
                print("eval_average_episode_rewards is {}.".format(np.mean(eval_episode_rewards)))
                break

    @torch.no_grad()
    def eval(self, step):
        eval_episode = 0
        eval_episode_rewards = []
        one_episode_rewards = []
        eval_episode_lengths = []
        one_episode_lengths = []
        for eval_i in range(self.n_eval_rollout_threads):
            one_episode_rewards.append([])
            one_episode_lengths.append(0)

        eval_obs, eval_share_obs, _ = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            self.trainer.prep_rollout()
            eval_actions, eval_rnn_states = \
                self.trainer.policy.act(np.concatenate(eval_share_obs.cpu().numpy()),
                                        np.concatenate(eval_obs.cpu().numpy()),
                                        np.concatenate(eval_rnn_states),
                                        np.concatenate(eval_masks),
                                        deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            # save now state and get next state
            eval_now_obs, now_eval_share_obs = eval_obs, eval_share_obs
            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, _, _ = self.eval_envs.step(torch.tensor(eval_actions.transpose(1, 0, 2)))
            eval_rewards = eval_rewards.cpu().numpy()
            eval_dones = eval_dones.cpu().numpy()

            # print('-----------------------------')
            # print('eval_obs', eval_obs.shape)
            # print('eval_share_obs', eval_share_obs.shape)
            # print('eval_rewards', eval_rewards.shape)
            # print('eval_rewards', eval_rewards[1])
            # all_episode_rewards += np.mean(eval_rewards, axis=1).flatten()
            # print('eval_dones', eval_dones.shape)
            # print('n_eval_rollout_threads', self.all_args.n_eval_rollout_threads)
            eval_dones_env = np.all(eval_dones, axis=1)
            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                one_episode_rewards[eval_i].append(np.mean(eval_rewards[eval_i]))
                one_episode_lengths[eval_i] += 1

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(np.sum(one_episode_rewards[eval_i]))
                    one_episode_rewards[eval_i] = []
                    eval_episode_lengths.append(np.sum(one_episode_lengths[eval_i]))
                    one_episode_lengths[eval_i] = 0

            if eval_episode >= self.all_args.eval_episodes:
                # eval_episode_rewards = torch.cat(eval_episode_rewards, dim=-1)
                # eval_episode_rewards = torch.cat(eval_episode_rewards)
                eval_env_infos = {
                    'eval_average_episode_rewards': np.mean(eval_episode_rewards),
                    'eval_max_episode_rewards': np.max(eval_episode_rewards),
                    'eval_average_episode_lengths': np.mean(eval_episode_lengths),
                    'eval_max_episode_lengthslengths': np.max(eval_episode_lengths),
                }
                print(eval_env_infos)
                print("eval_average_episode_rewards is {}.".format(np.mean(eval_episode_rewards)))
                print("eval_average_episode_lengths is {}.".format(np.mean(eval_episode_lengths)))
                break

    @torch.no_grad()
    def preference_eval(self, step):
        eval_episode = 0
        eval_episode_rewards = []
        one_episode_rewards = []
        eval_episode_lengths = []
        one_episode_lengths = []
        for eval_i in range(self.n_eval_rollout_threads):
            one_episode_rewards.append([])
            one_episode_lengths.append(0)

        ######################### add for preference reward
        reward_collector = RewardCollector(self.all_args)

        eval_obs, eval_share_obs, _ = self.eval_envs.reset()
        ######################### add for preference reward
        if self.all_args.use_preference_reward:
            self.pref_reward_assistant.his_next_obs[:, self.pref_reward_assistant.preference_traj_length - 1] = _t2n(eval_obs).copy()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            self.trainer.prep_rollout()
            eval_actions, eval_rnn_states = \
                self.trainer.policy.act(np.concatenate(eval_share_obs.cpu().numpy()),
                                        np.concatenate(eval_obs.cpu().numpy()),
                                        np.concatenate(eval_rnn_states),
                                        np.concatenate(eval_masks),
                                        deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            # save now state and get next state
            eval_now_obs, now_eval_share_obs = eval_obs, eval_share_obs
            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, _, _ = self.eval_envs.step(torch.tensor(eval_actions.transpose((1, 0, 2))))
            # eval_obs = _t2n(eval_obs)
            # eval_share_obs = _t2n(eval_share_obs)
            eval_rewards = eval_rewards.cpu().numpy()
            eval_dones = eval_dones.cpu().numpy()

            # print('-----------------------------')
            # print('eval_obs', eval_obs.shape)
            # print('eval_share_obs', eval_share_obs.shape)
            # print('eval_rewards', eval_rewards.shape)
            # print('eval_rewards', eval_rewards[1])
            # all_episode_rewards += np.mean(eval_rewards, axis=1).flatten()
            # print('eval_dones', eval_dones.shape)
            # print('n_eval_rollout_threads', self.all_args.n_eval_rollout_threads)
            eval_dones_env = np.all(eval_dones, axis=1)
            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                one_episode_rewards[eval_i].append(np.mean(eval_rewards[eval_i]))
                one_episode_lengths[eval_i] += 1

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(np.sum(one_episode_rewards[eval_i]))
                    one_episode_rewards[eval_i] = []
                    eval_episode_lengths.append(np.sum(one_episode_lengths[eval_i]))
                    one_episode_lengths[eval_i] = 0

            if self.all_args.use_preference_reward:
                ######################### preference: prefreence reward model
                pref_reward = self.pref_reward_assistant.insert(_t2n(eval_obs), eval_actions, eval_dones)
                self.pref_reward_assistant.clear_done()
                ######################### preference: insert (real, pref) pair
                reward_collector.insert(real_rewards=eval_rewards, pref_rewards=pref_reward, dones=eval_dones_env)

            if eval_episode >= self.all_args.eval_episodes:
                # eval_episode_rewards = torch.cat(eval_episode_rewards, dim=-1)
                # eval_episode_rewards = torch.cat(eval_episode_rewards)
                eval_env_infos = {
                    'eval_average_episode_rewards': np.mean(eval_episode_rewards),
                    'eval_max_episode_rewards': np.max(eval_episode_rewards),
                    'eval_average_episode_lengths': np.mean(eval_episode_lengths),
                    'eval_max_episode_lengthslengths': np.max(eval_episode_lengths),
                }
                print(eval_env_infos)
                print("eval_average_episode_rewards is {}.".format(np.mean(eval_episode_rewards)))
                print("eval_average_episode_lengths is {}.".format(np.mean(eval_episode_lengths)))
                reward_collector.save_data()
                break
