#!/bin/sh
env="mujoco"
scenario="HalfCheetah-v2"
agent_conf="6x1"
model_type="MultiPrefTransformerDivide"
agent_obsk=0
faulty_node=-1
#eval_faulty_node="-1 0 1 2 3 4 5"
eval_faulty_node="-1"
algo="mat"
exp="train_mujoco_halfcheetah-6x1_policy"
seed=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=0 python train_reward/train_reward_model.py \
--comment ${exp} \
--multi_transformer.embd_dim 256 --multi_transformer.n_layer 1 --multi_transformer.n_head 4 \
--multi_transformer.use_dropout True --multi_transformer.use_lstm True \
--batch_size 256 --n_epochs 10000 --seed ${seed} --model_type ${model_type} \
--dataset_path "/home/LAB/qiuyue/preference/preference_data/mujoco/"\
"halfcheetah-6x1_replaybuffer_5w_len100_dff10/preference_pair_data.pkl" \
--max_traj_length 100 --env "mujoco" --task "halfcheetah-6x1"

