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
CUDA_VISIBLE_DEVICES=0 python train_policy/train_mujoco.py --seed ${seed} --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
--scenario ${scenario} --agent_conf ${agent_conf} --agent_obsk ${agent_obsk} --faulty_node ${faulty_node} --eval_faulty_node ${eval_faulty_node} \
--critic_lr 5e-5 --lr 5e-5 --entropy_coef 0.001 --max_grad_norm 0.5 --eval_episodes 5 --n_training_threads 16 --n_rollout_threads 40 --num_mini_batch 40 \
--episode_length 100 --save_interval 5 --eval_interval 20 --log_interval 5 --num_env_steps 10000000 --ppo_epoch 10 --clip_param 0.05 --use_eval --add_center_xy \
--use_state_agent --use_value_active_masks --use_policy_active_masks --dec_actor \
--use_preference_reward --preference_model_type ${model_type} --preference_reward_std 0.1 \
--preference_model_dir "/home/LAB/qiuyue/preference/MAPT-master/mat/scripts/results/pref_reward/mujoco/halfcheetah-6x1/"\
"MultiPrefTransformerDivide/mujoco_halfcheetah-6x1_MPTD_embedDim256_encDecLstm_catObsAct_trajLen100_prefDiff10/models/reward_model_30.pt" \
--preference_embd_dim 256 --preference_n_layer 1 --preference_n_head 4 --preference_traj_length 100 \
--preference_use_dropout --preference_use_lstm

