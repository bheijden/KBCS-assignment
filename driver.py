import numpy as np
import time
import sys

class Driver:
    def __init__(self, env, replay_buffer=None):
        self.env = env
        self.replay_buffer = replay_buffer

        # CALLBACKS
        self.eps_counter = 0
        self.step_counter = 0

    def run(self, render=False, num_steps=None, logger=None):
        # todo: https://stackoverflow.com/questions/50107530/how-to-render-openai-gym-in-google-colab
        # Frame structure: (episode, action_k-1, cost_k, done, observation_k-1)
        #                      .             .         .    .          .
        #                      .             .         .    .          .
        #                      .             .         .    .          .
        #                  (episode,        NaN,      NaN, done, observation_k)

        
            # CHECK IF POSITIVE NUMBER OF STEPS
            if num_steps < 1:
                return

            # START BATCH
            it_t = 0
            while it_t < num_steps:
                eps_start = time.time()

                # RESET ENVIRONMENT AND POLICY
                observation, frame = self.env.reset()

                # ADD INITIAL OBSERVATION
                if self.replay_buffer is not None:
                    self.replay_buffer.add_episode()
                    self.replay_buffer.add(idx=0, obs=observation, state=frame['state'])

                # Assuming all zeros is the same as applying no action.
                # Might fail for discrete actions...!
                frame['action'] = self.env.action_space.sample()*0

                # START EPISODE
                t = 0
                cum_cost = 0
                while True:
                    if render:
                        self.env.render()

                    # SELECT ACTION
                    action = self.env.action_space.sample()

                    # RUN ENVIRONMENT
                    observation, cost, done, frame = self.env.step(action)

                    # ADD FRAME
                    if self.replay_buffer is not None:
                        self.replay_buffer.add(idx=t, obs=None, action=frame['action'], cost=cost)
                        self.replay_buffer.add(idx=t+1, obs=observation, state=frame['state'])

                    # CALLBACK
                    frame['cost'] = cost

                    cum_cost += cost
                    t += 1

                    if done:
                        it_t += t
                        if self.replay_buffer is not None:
                            self.replay_buffer.add(idx=t,
                                                   obs=None,
                                                   state=None,
                                                   action=np.full(self.env.action_space.sample().shape, np.nan,
                                                                  dtype=np.float32),
                                                   cost=np.array(np.nan, dtype=np.float32))
                            self.replay_buffer.remove_unfilled_steps(t)

                        # CALLBACK
                        frame['cost'] = np.array(np.nan, dtype=np.float32)

                        # LOG PERFORMANCE
                        self.step_counter += t
                        self.eps_counter += 1
                        eps_length = time.time() - eps_start
                        if logger is not None:
                            logger.log('train/duration', eps_length, self.step_counter)
                            logger.log('train/episode_step', t, self.step_counter)
                            logger.log('train/episode', self.eps_counter, self.step_counter)
                            logger.log('train/episode_cost', cum_cost, self.step_counter)
                            logger.log('train/episode_avg_cost', cum_cost / t, self.step_counter)
                            logger.dump(self.step_counter)
                        break
