#!/bin/sh
env="football"
scenario="academy_3_vs_1_with_keeper"
# academy_pass_and_shoot_with_keeper
# academy_3_vs_1_with_keeper
# academy_counterattack_easy
model_type="MultiPrefTransformerDivide"
n_agent=4
algo="mat"
exp="train_football_3_vs_1_policy"
seed=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=0 python train_reward/train_reward_model.py \
--comment ${exp} \
--multi_transformer.embd_dim 64 --multi_transformer.action_embd_dim 64 \
--multi_transformer.n_layer 1 --multi_transformer.n_head 4 \
--multi_transformer.use_dropout True --multi_transformer.use_lstm True \
--batch_size 256 --n_epochs 10000 --seed ${seed} --model_type ${model_type} \
--dataset_path "/home/LAB/qiuyue/preference/preference_data/football/"\
"academy_3_vs_1_replaybuffer_5w_len25_diff1/preference_pair_data.pkl" \
--max_traj_length 25 --env "football" --task ${scenario}

