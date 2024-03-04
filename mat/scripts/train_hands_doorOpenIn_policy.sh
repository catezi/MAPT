#!/bin/sh
env="hands"
task="ShadowHandDoorOpenInward"
model_type="MultiPrefTransformerDivide"
algo="happo"
exp="train_hands_doorOpenIn"
kl_threshold=0.016

echo "env is ${env}, task is ${task}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
CUDA_VISIBLE_DEVICES=0 python train_policy/train_hands_happo.py --env_name ${env} --task ${task} --algorithm_name ${algo} --experiment_name ${exp} \
--lr 5e-4 --critic_lr 5e-4 --std_x_coef 1 --std_y_coef 5e-1 --running_id 0 --n_rollout_threads 80 --num_mini_batch 1 \
--num_env_steps 20000000 --ppo_epoch 5 --kl_threshold ${kl_threshold} --use_value_active_masks \
--entropy_coef 0.0 --max_grad_norm 10 --gamma 0.96 --clip_param 0.2 --hidden_size 512 --layer_N 2 \
--log_interval 5 --save_interval 5 --eval_interval 10 --use_popart \
--use_preference_reward --preference_model_type ${model_type} --preference_reward_std 0.1 \
--preference_model_dir "/home/LAB/qiuyue/preference/MAPT-master/mat/scripts/results/pref_reward/hands/"\
"ShadowHandDoorOpenInward/MultiPrefTransformerDivide/train_hands_doorOpenIn/models/reward_model_510.pt" \
--preference_embd_dim 256 --preference_n_layer 1 --preference_n_head 4 \
--preference_config_length --preference_traj_length 50 --preference_use_dropout --preference_use_lstm --headless

