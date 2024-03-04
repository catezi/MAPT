import pickle
import numpy as np


def load_dataset(env, task, data_dir, action_type):
    with open(data_dir, "rb") as fp:
        dataset = pickle.load(fp)
    train_batch = {
        'observations0': [], 'observations1': [], 'actions0': [], 'actions1': [],
        'timesteps0': [], 'timesteps1': [], 'labels': [],
    }
    val_batch = {
        'observations0': [], 'observations1': [], 'actions0': [], 'actions1': [],
        'timesteps0': [], 'timesteps1': [], 'labels': [],
    }
    # split train and val dataset
    train_traj_num = int(0.8 * len(dataset))
    for i, path in enumerate(dataset):
        # must ensure all len indataset is max_len
        max_len = path['traj0']['obs'].shape[0]
        if path['traj0']['obs'].shape[0] < max_len:
            continue
        if i < train_traj_num:
            # print('actions', path['traj0']['actions'][:max_len].shape)
            train_batch['observations0'].append(np.expand_dims(path['traj0']['obs'][:max_len], axis=0))
            train_batch['actions0'].append(np.expand_dims(path['traj0']['actions'][:max_len], axis=0))
            train_batch['timesteps0'].append(np.expand_dims(np.arange(0, max_len), axis=0))
            train_batch['observations1'].append(np.expand_dims(path['traj1']['obs'][:max_len], axis=0))
            train_batch['actions1'].append(np.expand_dims(path['traj1']['actions'][:max_len], axis=0))
            train_batch['timesteps1'].append(np.expand_dims(np.arange(0, max_len), axis=0))
            train_batch['labels'].append(
                np.expand_dims(np.array([1.0, 0.0] if path['label'] == 1 else (
                    [0.0, 1.0] if path['label'] == -1 else [0.5, 0.5]
                )), axis=0)
            )
        else:
            val_batch['observations0'].append(np.expand_dims(path['traj0']['obs'][:max_len], axis=0))
            val_batch['actions0'].append(np.expand_dims(path['traj0']['actions'][:max_len], axis=0))
            val_batch['timesteps0'].append(np.expand_dims(np.arange(0, max_len), axis=0))
            val_batch['observations1'].append(np.expand_dims(path['traj1']['obs'][:max_len], axis=0))
            val_batch['actions1'].append(np.expand_dims(path['traj1']['actions'][:max_len], axis=0))
            val_batch['timesteps1'].append(np.expand_dims(np.arange(0, max_len), axis=0))
            val_batch['labels'].append(
                np.expand_dims(np.array([1.0, 0.0] if path['label'] == 1 else (
                    [0.0, 1.0] if path['label'] == -1 else [0.5, 0.5]
                )), axis=0)
            )
    for key in train_batch:
        train_batch[key] = np.concatenate(train_batch[key], axis=0)
        val_batch[key] = np.concatenate(val_batch[key], axis=0)
        print(key, train_batch[key].shape)
        print(key, val_batch[key].shape)
    # set game info
    env_info = {
        'observation_dim': train_batch['observations0'].shape[-1],
        'action_dim': train_batch['actions0'].shape[-1]
        if action_type == 'Continous'
        else all_discrete_action_num[env][task],
        'max_len': max_len,
        'n_agent': train_batch['observations0'].shape[-2],
    }

    return train_batch, val_batch, env_info


all_discrete_action_num = {
    'smac': {
        '3m': 9,
        '3s5z': 14,
        '6h_vs_8z': 14,
        'MMM2': 18,
    },
    'football': {
        'academy_pass_and_shoot_with_keeper': 19,
        'academy_3_vs_1_with_keeper': 19,
        'academy_counterattack_easy': 19,
    },
}

