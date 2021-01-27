import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from gym.envs.registration import register
import skimage.transform

register(
    id='pendulum-underactuated-v0',
    entry_point='environments:UnderactuatedPendulum',
    max_episode_steps=200,
)

class UnderactuatedPendulum(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, g=10.0, render_shape=(500, 500), model=None):
        self.max_speed = 8.  # 8
        self.max_torque = 2.
        self.dt = .05
        self.g = g
        self.m = 1.
        self.l = 1.
        self.viewer = None

        high = np.array([1., 1., self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque, shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )
        
        high_state = np.array([np.pi, self.max_speed], dtype=np.float32)
        self.state_space = spaces.Box(
            low=-high_state,
            high=high_state,
            dtype=np.float32
        )

        self.seed()
        self.render_shape = render_shape
        self.render_init = False
        self.model = model

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, False, {"state": np.array([angle_normalize(newth), newthdot]), 'action': self.last_u}

    def reset(self):
        high = np.array([np.pi, 1])
        low = -high
        self.state = self.np_random.uniform(low=low, high=high)
        self.last_u = None
        return self._get_obs(), {"state": self.state}

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode='human'):
        if not self.render_init:
            self.render_init = True
            from gym.envs.classic_control import rendering
            # rod
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            # axle
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)

            self.viewer = rendering.Viewer(*self.render_shape)
            self.viewer.set_bounds(-1.2, 1.2, -1.2, 1.2)
            self.viewer.add_geom(rod)
            if self.model is not None:
                zaxis = rendering.Line((0, 0), (0, 1.2))
                th_axis = rendering.Line((0, 0), (1.2, 0))
                self.pole_pred_transform = rendering.Transform()
                th_axis.add_attr(self.pole_pred_transform)
                self.viewer.add_geom(zaxis)
                self.viewer.add_geom(th_axis)
            else:
                self.pole_pred_transform = None
            self.viewer.add_geom(axle)

        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        
        if self.pole_pred_transform is not None:
            image = self.viewer.render(return_rgb_array=True)
            kwargs = dict(output_shape=(28, 28), mode='edge', order=1, preserve_range=True)
            observation = skimage.transform.resize(image, **kwargs).astype(np.float32) / 255
            model_output = self.model.predict(observation[None, :])[0]
            if len(model_output) == 1:
                th_pred = model_output[0]
            elif len(model_output) == 2:
                th_pred = np.arctan2(model_output[0], model_output[1])
            else:
                raise ValueError('Model output not compatible with simulator. Model should only output [theta] or [sin(theta), cos(theta)] prepended with the batch dimension.')
            self.pole_pred_transform.set_rotation(th_pred + np.pi / 2)
        
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        self.render_init = False
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
