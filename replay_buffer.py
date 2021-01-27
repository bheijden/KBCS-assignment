import numpy as np
from collections import deque
import os
import h5py
import psutil

class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, state_shape, num_episodes, len_time):
        self.buffer = deque(maxlen=num_episodes)
        self.num_episodes = num_episodes
        self.len_time = len_time
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.state_shape = state_shape

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        self.obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8
        self.key = 'observation' if len(obs_shape) == 1 else 'image'

        self.idx = 0
        self.last_save = 0
        self.full = False

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, index):
        return self.buffer[index]

    def add_episode(self, eps_buf=None):
        if eps_buf is None:
            eps_buf = {'observation': np.empty((self.len_time + 1, *self.obs_shape), dtype=self.obs_dtype),
                       'action': np.empty((self.len_time + 1, *self.action_shape), dtype=np.float32),
                       'state': np.empty((self.len_time + 1, *self.state_shape), dtype=np.float32),
                       'cost': np.empty((self.len_time + 1, 1), dtype=np.float32)}
        self.buffer.append(eps_buf)
        self.idx = (self.idx + 1) % self.num_episodes
        self.full = self.full or self.idx == 0

    def add(self, idx, obs=None, action=None, cost=None, state=None):
        if obs is not None:
            np.copyto(self.buffer[-1]['observation'][idx], obs[self.key])
        if action is not None:
            np.copyto(self.buffer[-1]['action'][idx], action)
        if cost is not None:
            np.copyto(self.buffer[-1]['cost'][idx], cost)
        if state is not None:
            np.copyto(self.buffer[-1]['state'][idx], state)

    def remove_unfilled_steps(self, t_max):
        if t_max + 1 == self.len_time + 1:
            return
        for key, value in self.buffer[-1].items():
            self.buffer[-1][key] = value[:t_max + 1]

    def save_buffer(self, dir, name_dataset=None):
        if name_dataset is None:
            name_dataset = 'dataset'
        if not os.path.exists(dir):
            os.makedirs(dir)
        
        data_obs = []
        data_state = []
        
        for eps in self.buffer:
            # data_obs.append(np.transpose(eps['observation'], axes=[1, 2, 3, 0]))
            # data_state.append(np.transpose(eps['state'], axes=[1, 0]))
            data_obs.append(eps['observation'])
            data_state.append(eps['state'])
        
        data_obs = np.concatenate(data_obs, axis=0)
        data_state = np.concatenate(data_state, axis=0)

        file_name = '%s/%s.h5' % (dir,  name_dataset)
        with h5py.File(file_name, 'w') as hf:
            hf.create_dataset('observation.h5', data=data_obs)
            hf.create_dataset('state.h5', data=data_state)
        print('Dataset saved at %s' % file_name)