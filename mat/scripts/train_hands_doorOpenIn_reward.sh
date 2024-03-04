#!/bin/sh
env="hands"
task="ShadowHandDoorOpenInward"
model_type="MultiPrefTransformerDivide"
algo="happo"
exp="train_hands_doorOpenIn"
seed=1

echo "env is ${env}, task is ${task}, algo is ${algo}, exp is ${exp}"
CUDA_VISIBLE_DEVICES=0 python train_reward/train_reward_model.py \
--comment ${exp} \
--multi_transformer.embd_dim 256 --multi_transformer.n_layer 1 --multi_transformer.n_head 4 \
--multi_transformer.use_dropout True --multi_transformer.use_lstm True \
--batch_size 256 --n_epochs 1000 --seed ${seed} --model_type ${model_type} \
--dataset_path "/home/LAB/qiuyue/preference/preference_data/hands/"\
"ShadowHandDoorOpenInward_replaybuffer_5w_len50_diff10/preference_pair_data.pkl" \
--max_traj_length 50 --env "hands" --task ${task}


