import numpy as np
import gym
import wrappers
from driver import Driver
from logger import Logger
from replay_buffer import ReplayBuffer
from support_fns import seed_experiment
import colorama
from environments.pendulum import UnderactuatedPendulum

# INITIALIZE COLORS FOR PROMPT ON WIN10 MACHINES
colorama.init()

# DATASET PARAMETERS
max_num_episodes = 120
len_time = 100
render_shape = (500, 500)
image_res = (28, 28)

# INITIALIZE PENDULUM ENVIRONMENT
env = gym.make('pendulum-underactuated-v0', render_shape=render_shape, model=None)
env = wrappers.NormalizeActions(env)
env = wrappers.MinimumDuration(env, len_time)
env = wrappers.MaximumDuration(env, len_time)
env = wrappers.ObservationDict(env, key='observation')
env = wrappers.PixelObservations(env, image_res, np.uint8, 'image')
env = wrappers.ConvertRewardToCost(env)
env = wrappers.ConvertTo32Bit(env)

# SEED EXPERIMENT TO CREATE REPRODUCIBLE RESULTS
seed_value = 0
seed_experiment(seed=seed_value)
env.seed(seed=seed_value)

# GET ENVIRONMENT DATA SHAPES
observation_shape = env.observation_space['image'].shape
action_shape = env.action_space.shape
state_shape = env.state_space.shape

# INITIALIZE INFRASTRUCTURE
logger = Logger('.')
replay_buffer = ReplayBuffer(observation_shape, action_shape, state_shape, max_num_episodes, len_time)
driver = Driver(env, replay_buffer=replay_buffer)

# GATHER EXPERIENCE
print('Generating dataset. Generate %d episodes of length %d.' % (max_num_episodes, len_time + 1))
driver.run(render=True, num_steps=max_num_episodes*len_time, logger=logger)

# SAVE DATASET
replay_buffer.save_buffer('.', name_dataset='dataset')