# Multi-agent Preference Transformer (MAPT)

This is the official code repository for the paper "Decoding Global Preferences: Temporal and Cooperative Dependency Modeling in Multi-Agent Preference-Based Reinforcement Learning" accepted by AAAI 2024.

## Installation

### 1. Installing Dependences

``` Bash
pip install -r requirements.txt
```

#### Multi-agent MuJoCo

Following the instructios in https://github.com/openai/mujoco-py and https://github.com/schroederdewitt/multiagent_mujoco to setup a mujoco environment. In the end, remember to set the following environment variables:

``` Bash
LD_LIBRARY_PATH=${HOME}/.mujoco/mujoco200/bin;
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
```

#### StarCraft II & SMAC

Run the script

``` Bash
bash install_sc2.sh
```

Or you could install them manually to other path you like, just follow here: https://github.com/oxwhirl/smac.

#### Google Research Football

Please following the instructios in https://github.com/google-research/football. 

#### Bi-DexHands 

Please following the instructios in https://github.com/PKU-MARL/DexterousHands. 


### 2. Preparing Preference Data

Following our paper, we use script teachers to generate preference data.

Please download demonstrate preference data of several tasks from [here](https://bhpan.buaa.edu.cn/link/AAFAAAA5703BEA491E903A81CF6715F7C4).


### 3. Reward Modeling Phrase

- After installing dependences, you could run shells in the "scripts" folder.
- Run scripts end with `"reward.sh"`, for example `"train_smac_3m_reward.sh"`.
- Remember to set `--dataset_path` correctly to the path of your data set.
- Remember to set `log_dir` to the path you want to save log and model, which is set `'./results/pref_reward'` by defaught.

### 4. Policy Learning Phrase

- Run scripts end with `"policy.sh"`, for example `"train_smac_3m_policy.sh"`.
- Remember to set `--preference_model_dir` correctly to the path of pre-trained preference reward model.

### 5. Parameter Description

- parameters for the reward modeling phrase

| Parameter name                    | Description of parameter                                     |
| --------------------------------- | ------------------------------------------------------------ |
| comment                           | Tag for your experiment, which will be appended to the `log_dir` |
| multi_transformer.embd_dim        | Dimension of model                                           |
| multi_transformer.action_embd_dim | Dimension of action embedding                                |
| multi_transformer.n_layer         | Number of attention layer                                    |
| multi_transformer.n_head          | Number of attention head                                     |
| multi_transformer.use_dropout     | If use drop out                                              |
| multi_transformer.use_lstm        | If use lstm for time layer                                   |
| batch_size                        | Batch size of training input data                            |
| n_epochs                          | Number of epochs for training                                |
| seed                              | Seed for training                                            |
| model_type                        | Reward model type, defaults to `MultiPrefTransformerDivide`(our model) |
| dataset_path                      | Path for dataset                                             |
| max_traj_length                   | Trajectory length used for training (not longer than origin trajectory length in dataset) |
| env、task                         | environment(benchmark)  and task for training data           |

- parameters for the policy learning phrase (Remember that we did not explain parameter for `mat` and `happo` algorithm, please refer to repository of [mat](https://github.com/PKU-MARL/Multi-Agent-Transformer) and [happo](https://github.com/cyanrain7/TRPO-in-MARL))

| Parameter name         | Description of parameter                                     |
| ---------------------- | ------------------------------------------------------------ |
| use_preference_reward  | If use preference reward form preference reward model(always true in our experiment) |
| preference_model_type  | Reward model type, defaults to `MultiPrefTransformerDivide`(our model) |
| preference_reward_std  | Standard deviation for preference reward normalization       |
| preference_model_dir   | Path of pre-trained preference model                         |
| preference_embd_dim    | Dimension of preference model                                |
| preference_n_layer     | Number of attention layer for preference model               |
| preference_n_head      | Number of attention head for preference model                |
| preference_traj_length | Number of trajectory length used for training preference model |
| preference_use_dropout | If use drop out for training preference model                |
| preference_use_lstm    | If use lstm for time layer of preference model               |

