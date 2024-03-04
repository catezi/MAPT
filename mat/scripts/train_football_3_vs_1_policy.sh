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
CUDA_VISIBLE_DEVICES=0 python train_policy/train_football.py --seed ${seed} --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario ${scenario} --n_agent ${n_agent} \
--lr 5e-4 --entropy_coef 0.01 --max_grad_norm 0.5 --eval_episodes 32 --n_training_threads 16 --n_rollout_threads 20 --num_mini_batch 1 --episode_length 200 --eval_interval 25 \
--num_env_steps 10000000 --ppo_epoch 10 --clip_param 0.05 --use_eval --use_value_active_masks --use_policy_active_masks \
--save_interval 5 -log_interval 5 --eval_interval 20 \
--use_preference_reward --preference_model_type ${model_type} --preference_reward_std 0.1 \
--preference_model_dir "/home/LAB/qiuyue/preference/MAPT-master/mat/scripts/results/pref_reward/football/academy_3_vs_1_with_keeper/MultiPrefTransformerDivide/"\
"football_academy_3_vs_1_with_keeper_MPTD_embedDim64_encDecLstm_catObsAct_prefDiff1_trajLen25/models/reward_model_85.pt" \
--preference_embd_dim 64 --preference_n_layer 1 --preference_n_head 4 \
--preference_traj_length 100 --preference_use_dropout --preference_use_lstm

