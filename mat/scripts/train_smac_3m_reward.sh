#!/bin/sh
env="StarCraft2"
map="3m"
model_type="MultiPrefTransformerDivide"
exp="train_smac_3m"
seed=1

echo "env is ${env}, task is ${map}, model_type is ${model_type}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=0 python train_reward/train_reward_model.py \
--comment ${exp} \
--multi_transformer.embd_dim 256 --multi_transformer.action_embd_dim 64 \
--multi_transformer.n_layer 1 --multi_transformer.n_head 4 \
--multi_transformer.use_dropout True --multi_transformer.use_lstm True \
--batch_size 256 --n_epochs 1000 --seed ${seed} --model_type ${model_type} \
--dataset_path "/home/LAB/qiuyue/preference/preference_data/smac/"\
"3m_replaybuffer_5w_len16_diff0.5/preference_pair_data.pkl" \
--max_traj_length 16 --env "smac" --task ${map}






