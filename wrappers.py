import numpy as np
import gym
import skimage.transform
import nested


class NormalizeActions(object):
    """
    if and(low, high) not np.inf: scales action from [-1, 1] back to the unnormalized action
    if or(low,high) np.inf: no normalization of the actions, and true action must be used"""

    def __init__(self, env):
        self._env = env
        low, high = env.action_space.low, env.action_space.high
        self._enabled = np.logical_and(np.isfinite(low), np.isfinite(high))
        self._low = np.where(self._enabled, low, -np.ones_like(low))
        self._high = np.where(self._enabled, high, np.ones_like(low))

        # Check if normalized actions are centered around 0
        nonzeros = (self._low + self._high).any()
        if nonzeros:
          raise ValueError('Bounded action of environment not centered around zero!')

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def action_space(self):
        space = self._env.action_space
        low = np.where(self._enabled, -np.ones_like(space.low), space.low)
        high = np.where(self._enabled, np.ones_like(space.high), space.high)
        return gym.spaces.Box(low, high, dtype=space.dtype)

    def step(self, action):
        # de-normalize action
        action = (action + 1) / 2 * (self._high - self._low) + self._low

        # apply action
        obs, reward, done, info = self._env.step(action)

        # normalize applied action (in case action was above maximum)
        info['action'] = (2*info['action'] - (self._high + self._low))/(self._high - self._low)
        return obs, reward, done, info

    
class MinimumDuration(object):
    """Extends the episode to a given lower number of decision points."""

    def __init__(self, env, duration):
        self._env = env
        self._duration = duration
        self._step = None

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        observ, reward, done, info = self._env.step(action)
        self._step += 1
        if self._step < self._duration:
            done = False
        return observ, reward, done, info

    def reset(self):
        self._step = 0
        return self._env.reset()
    
    
class MaximumDuration(object):
    """Limits the episode to a given upper number of decision points."""

    def __init__(self, env, duration):
        self._env = env
        self._duration = duration
        self._step = None

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        if self._step is None:
            raise RuntimeError('Must reset environment.')
        observ, reward, done, info = self._env.step(action)
        self._step += 1
        if self._step >= self._duration:
            done = True
            self._step = None
        return observ, reward, done, info

    def reset(self):
        self._step = 0
        return self._env.reset()
    

class ObservationDict(object):
    def __init__(self, env, key='observ'):
        self._env = env
        self._key = key

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def observation_space(self):
        spaces = {self._key: self._env.observation_space}
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        return self._env.action_space

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs = {self._key: np.array(obs)}
        return obs, reward, done, info

    def reset(self):
        obs, info = self._env.reset()
        obs = {self._key: np.array(obs)}
        return obs, info


class PixelObservations(object):

    def __init__(self, env, size=(64, 64), dtype=np.uint8, key='image'):
        assert isinstance(env.observation_space, gym.spaces.Dict)
        self._env = env
        self._size = size
        self._dtype = dtype
        self._key = key

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def observation_space(self):
        high = {np.uint8: 255, np.float32: 1.0, np.float64: 1.0}[self._dtype]
        image = gym.spaces.Box(0, high, self._size + (3,), dtype=self._dtype)
        spaces = self._env.observation_space.spaces.copy()
        assert self._key not in spaces
        spaces[self._key] = image
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        return self._env.action_space

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        # self._obs = [self._render_image()] + self._obs[:-1]
        # obs[self._key] = np.stack(self._obs, axis=0)
        obs[self._key] = self._render_image()
        return obs, reward, done, info

    def reset(self):
        obs, info = self._env.reset()
        # self._obs = [self._render_image()] * self._stack_images
        # obs[self._key] = np.stack(self._obs, axis=0)
        obs[self._key] = self._render_image()
        return obs, info

    def _render_image(self):
        image = self._env.render('rgb_array')
        if image.shape[:2] != self._size:
            kwargs = dict(
                output_shape=self._size, mode='edge', order=1, preserve_range=True)
            image = skimage.transform.resize(image, **kwargs).astype(image.dtype)
        if self._dtype and image.dtype != self._dtype:
            if image.dtype in (np.float32, np.float64) and self._dtype == np.uint8:
                image = (image * 255).astype(self._dtype)
            elif image.dtype == np.uint8 and self._dtype in (np.float32, np.float64):
                image = image.astype(self._dtype) / 255
            else:
                message = 'Cannot convert observations from {} to {}.'
                raise NotImplementedError(message.format(image.dtype, self._dtype))
        return image
    

class ConvertTo32Bit(object):
    """Convert data types of an OpenAI Gym environment to 32 bit."""

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        observ, reward, done, info = self._env.step(action)
        observ = nested.map(self._convert_observ, observ)
        info = nested.map(self._convert_observ, info)
        reward = self._convert_reward(reward)
        return observ, reward, done, info

    def reset(self):
        observ, info = self._env.reset()
        observ = nested.map(self._convert_observ, observ)
        info = nested.map(self._convert_observ, info)
        return observ, info

    def _convert_observ(self, observ):
        if not np.isfinite(observ).all():
            raise ValueError('Infinite observation encountered.')
        if isinstance(observ, bool):
            return observ
        if observ.dtype == np.float64:
            return observ.astype(np.float32)
        elif observ.dtype == np.int64:
            return observ.astype(np.int32)
        return observ

    def _convert_reward(self, reward):
        if not np.isfinite(reward).all():
            raise ValueError('Infinite reward encountered.')
        return np.array(reward, dtype=np.float32)
    

class ConvertRewardToCost(object):

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        cost = -reward
        return obs, cost, done, info