#!/bin/sh
env="StarCraft2"
map="3m"
model_type="MultiPrefTransformerDivide"
algo="mat"
exp="train_smac_3m"
seed=1

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=0 python train_policy/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} --seed ${seed} \
--n_training_threads 16 --n_rollout_threads 32 --num_mini_batch 1 --episode_length 100 --num_env_steps 10000000 --lr 5e-4 --ppo_epoch 15 \
--clip_param 0.2 --save_interval 10 --use_value_active_masks --use_eval --save_interval 5 -log_interval 5 --eval_interval 20 \
--use_preference_reward --preference_model_type ${model_type} --preference_reward_std 0.1 \
--preference_model_dir "/home/LAB/qiuyue/preference/MAPT-master/mat/scripts/results/pref_reward/smac/3m/MultiPrefTransformerDivide/"\
"smac_3m_MPTD_embedDim3_encDecLstm_catObsAct_trajNum5w_trajLen32/models/reward_model_15.pt" \
--preference_embd_dim 256 --preference_n_layer 1 --preference_n_head 4 \
--preference_traj_length 100 --preference_use_dropout --preference_use_lstm


