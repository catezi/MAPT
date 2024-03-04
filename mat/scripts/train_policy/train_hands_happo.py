#!/usr/bin/env python
import sys
import os
import socket
import setproctitle
import numpy as np
from pathlib import Path
import isaacgym
import torch
import yaml
sys.path.append("../..")
from mat.config import get_config
from mat.envs.dexteroushandenvs.utils.config import get_args, parse_sim_params, load_env_cfg
from mat.envs.dexteroushandenvs.utils.parse_task import parse_task
from mat.envs.dexteroushandenvs.utils.process_marl import get_AgentIndex
from mat.runner.separated.hands_runner import HandsRunner as Runner


def make_train_env(all_args):
    if all_args.env_name == "hands":
        args = get_args(all_args=all_args)
        cfg = load_env_cfg(args)
        cfg["env"]["numEnvs"] = all_args.n_rollout_threads
        all_args.episode_length = cfg["env"]["episodeLength"]
        sim_params = parse_sim_params(args, cfg)
        agent_index = get_AgentIndex(cfg)
        env = parse_task(args, cfg, sim_params, agent_index)

        return env
    else:
        raise NotImplementedError


def make_eval_env(all_args):
    if all_args.env_name == "hands":
        args = get_args(all_args=all_args)
        cfg = load_env_cfg(args)
        cfg["env"]["numEnvs"] = all_args.n_eval_rollout_threads
        # all_args.n_eval_rollout_threads = all_args.eval_episodes
        sim_params = parse_sim_params(args, cfg)
        agent_index = get_AgentIndex(cfg)
        env = parse_task(args, cfg, sim_params, agent_index)
        return env
    else:
        raise NotImplementedError


def parse_args(args, parser):
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--add_move_state", action='store_true', default=False)
    parser.add_argument("--add_local_obs", action='store_true', default=False)
    parser.add_argument("--add_distance_state", action='store_true', default=False)
    parser.add_argument("--add_enemy_action_state", action='store_true', default=False)
    parser.add_argument("--add_agent_id", action='store_true', default=False)
    parser.add_argument("--add_visible_state", action='store_true', default=False)
    parser.add_argument("--add_xy_state", action='store_true', default=False)
    parser.add_argument("--run_dir", type=str, default='')

    # agent-specific state should be designed carefully
    parser.add_argument("--use_state_agent", action='store_true', default=False)
    parser.add_argument("--use_mustalive", action='store_false', default=True)
    parser.add_argument("--add_center_xy", action='store_true', default=False)
    parser.add_argument("--use_single_network", action='store_true', default=False)

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    print("all config: ", all_args)
    if all_args.seed_specify:
        all_args.seed = all_args.running_id
    else:
        all_args.seed = np.random.randint(1000, 10000)
    print("seed is :", all_args.seed)
    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    if all_args.config_dir:
        run_dir = Path(all_args.log_dir + "/results") / all_args.env_name / all_args.task / all_args.algorithm_name / all_args.experiment_name
    else:
        run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                           0] + "/results") / all_args.env_name / all_args.task / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if not run_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
            all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    if all_args.preference_eval:
        all_args.use_eval = True
    # env
    envs = make_train_env(all_args) if not all_args.use_eval else None
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    envs = eval_envs if all_args.use_eval else envs
    num_agents = envs.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    runner = Runner(config)
    if all_args.preference_eval:
        runner.preference_eval()
    else:
        runner.run()

    runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
    runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
