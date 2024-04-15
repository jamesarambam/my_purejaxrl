# ================================ Imports ================================ #
import numpy as np
from pdb import set_trace
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
# plt.ion()
from matplotlib.patches import Rectangle

# =============================== Variables ================================== #


# ============================================================================ #


import gym
from gym import spaces
from gym.spaces.space import Space
import logging

import time
# from parameters import SGW_WIDTH, SGW_HEIGHT

from gym.envs.classic_control import rendering

logger = logging.getLogger(__name__)

class SafeGridWorld_Gym(gym.Env):
    """Custom Environment that follows gym interface."""
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 10}

    def __init__(self, grid_size=(4, 4), horizon=20, transition_prob=0.8, width=10, height=5, seed=None):

        """
        This is the safe grid world in standard gym format.
        :param gridSize:        size of grid world in tuple(gridWidth, gridHeight).
        :param frisbee_state:   final goal states in list.
        :param holes_state:     holes states in list
        :param init_state:      init state for agent to start. Random position if "None".
        :param transition_prob: transition probability of env
        :param max_steps:       max steps that agent travel in env. set none if no limit.

        Default Map:
            # | A |   |   |   |       | 12| 13| 14| 15|
            # |   | 0 |   | 0 |       | 8 | 9 | 10| 11|
            # |   |   |   |   |       | 4 | 5 | 6 | 7 |
            # | 0 |   |   | x |       | 0 | 1 | 2 | 3 |
        """

        md = map_details(grid_size=grid_size)
        frisbee_state = md.frisbee_state
        holes_state = md.holes_state
        init_state = md.init_state
        max_steps = horizon

        # Standard Gym Requirement
        # Define action and observation space
        self.action_space = spaces.Discrete(4)
        self.action_space.seed(seed)

        self.num_class = grid_size[0] * grid_size[0]

        # self.obs_dtype = np.float32
        # self.observation_space = spaces.Box(low=float('-inf'), high=float('inf'), shape=(1,), dtype=self.obs_dtype)

        # self.obs_dtype = int
        # self.observation_space = spaces.Box(low=float('-inf'), high=float('inf'), shape=(1,), dtype=self.obs_dtype)
        self.observation_space = spaces.Discrete(self.num_class)

        # self.observation_space = spaces.MultiDiscrete([2 for _ in range(self.num_class)])
        # self.observation_space = OneHotEncoding(size=self.num_class)


        self.reward_range = (-float("inf"), float("inf"))

        # Environment information
        self.max_steps = max_steps
        self.num_step = 0
        self.episode = 0
        self.width = width
        self.height = height

        # Load the parameters
        self.gridWidth = grid_size[0]
        self.gridHeight = grid_size[1]
        # Environment configuration
        self.gridNum = self.gridWidth * self.gridHeight
        self.states = range(self.gridNum)
        if frisbee_state is None and holes_state is None:
            frisbee_state = [3]
            holes_state = [9, 11, 0]
        elif not (frisbee_state is not None and holes_state is not None):
            raise ValueError("Set the 'frisbee_state' and 'holes_state' simultaneously.")
        self.frisbee_states = frisbee_state
        self.holes_states = holes_state
        if init_state is None:
            self.init_state = None
            while True:
                self.state = np.random.choice(self.states)
                if (self.state not in self.frisbee_states) and (self.state not in self.holes_states):
                    break
            self.init_random = True
        else:
            self.init_state = init_state
            self.init_random = False
        self.transition_prob = transition_prob

        # Bound configuration
        #   get the boundary state using grid confiuration
        self.upBound = range(self.gridNum - self.gridWidth, self.gridNum)
        self.rightBound = []
        for i in range(self.gridHeight):
            self.rightBound.append((self.gridWidth - 1) + i * self.gridWidth)
        self.downBound = range(self.gridWidth)
        self.leftBound = []
        for i in range(self.gridHeight):
            self.leftBound.append(0 + i * self.gridWidth)

        # Action space
        #   create action dictionary, save the action and the change value of the state
        self.actionsDic = {0: self.gridWidth, 1: 1, 2: -self.gridWidth, 3: -1}
        # self.action_space = 5
        # self.actions = ['up', 'right', 'down', 'left', 'idle']
        # self.actionsDic = {'up': self.gridWidth, 'right': 1, 'down': -self.gridWidth, 'left': -1, 'idle': 0}

        # Reward Function
        #   save the reward for each action in array
        self.reward_function = np.zeros(self.gridNum)
        goal_reward = 50
        hole_reward = -5
        self.step_cost = -1  # Penalty for staying
        for state in self.frisbee_states:
            self.reward_function[state] = goal_reward
        for state in self.holes_states:
            self.reward_function[state] = hole_reward

        # Screen Configuration
        self.screen_width = (self.gridWidth + 2) * 100
        self.screen_height = (self.gridHeight + 2) * 100

        # Agent information
        self.gamma = 0.9
        self.state = None

        # Gym viewering
        self.viewer = None
        self.is_render = False
        self.is_sleep = False

    def one_hot(self, a):
        return np.squeeze(np.eye(self.num_class)[a.reshape(-1)])

    def step(self, action):

        # observation = np.resize(np.array(self.state, dtype=self.obs_dtype), new_shape=(1,))
        observation = np.resize(np.array(self.state), new_shape=(1,))

        reward = 0
        is_terminal = False
        info = {'goal': True, 'max_step': False}
        info['cur_state'] = self.state

        # Check env
        #   if env is in the terminal state, return.
        #   in case star in terminal state or did not break out
        if self.state in self.frisbee_states:
            is_terminal = True
            info['goal'] = True

            return observation, reward, is_terminal, {'goal': True}

        # elif self.state in self.holes_states:
        #     is_terminal = True
        #     return observation, reward, is_terminal, {'goal': False}

        # State transition
        #   transfer the state according to action;
        #   if action out of the bound, then set the state back.
        # deterministic and non-deterministic
        state = self.state
        p = np.random.uniform()
        # if random.uniform(0, 1) < self.transition_prob:
        if p < self.transition_prob:
            next_state = state + self.actionsDic[action]
        else:
            # equal possibility for other actions
            _action = action
            while _action == action:
                _action = self.action_space.sample()
            action = _action
            next_state = state + self.actionsDic[action]

        # consider Boundary
        if action == 0 and state in self.upBound:
            next_state = state
        elif action == 1 and state in self.rightBound:
            next_state = state
        elif action == 2 and state in self.downBound:
            next_state = state
        elif action == 3 and state in self.leftBound:
            next_state = state
        self.state = next_state

        # Reward and Cost
        cost = 0
        if next_state in self.holes_states:
            cost = self.reward_function[next_state]
        else:
            cost = 0

        if next_state in self.frisbee_states:
            reward = self.reward_function[next_state]
        else:
            reward = 0
        reward += self.step_cost

        # Terminal state
        is_terminal = False
        info['goal'] = False
        if next_state in self.frisbee_states:
            is_terminal = True
            info['goal'] = True
        # elif next_state in self.holes_states:
        #     is_terminal = True

        info['cost'] = cost

        # Render
        # if self.is_render:
        #     self.render()
        #     if self.is_sleep:
        #         time.sleep(.1)
        self.num_step += 1
        if self.num_step == self.max_steps - 1:
            is_terminal = True
            info['max_step'] = True

        # observation = np.resize(np.array(self.state, dtype=self.obs_dtype), new_shape=(1,))

        # observation = np.resize(np.array(self.state), new_shape=(1,))

        info['num_steps'] = self.num_step

        obs_onehot = self.one_hot(np.array([self.state]))

        return obs_onehot, reward, is_terminal, info

        # obs_onehot = self.one_hot(np.array([self.state]))
        # return obs_onehot, reward, is_terminal, info

    def reset(self):
        self.episode += 1
        self.num_step = 0

        self.reset_called = True
        # print("episode", self.episode)
        observation = None

        if self.init_random:
            while True:
                self.state = np.random.choice(self.states)
                if (self.state not in self.frisbee_states) or (self.state not in self.holes_states):
                    break
        else:
            self.state = self.init_state


        # observation = np.resize(np.array(self.state, dtype=self.obs_dtype), new_shape=(1,))
        # observation = np.resize(np.array(self.state), new_shape=(1,))
        # observation = np.array([self.state])

        # set_trace()
        obs_onehot = self.one_hot(np.array([self.state]))

        # print(observation)
        # return observation
        # return self.state
        return obs_onehot  # reward, done, info can't be included

    def render(self, mode='human'):
        # Initial render setting
        if self.viewer is None:
            self.viewer = rendering.Viewer(self.screen_width, self.screen_height-90)

            # Create the grid
            self.lines = []
            for i in range(1, self.gridHeight + 2):
                self.lines.append(rendering.Line((100, i * 100), ((self.gridWidth + 1) * 100, i * 100)))
            for i in range(1, self.gridWidth + 2):
                self.lines.append(rendering.Line((i * 100, 100), (i * 100, (self.gridHeight + 1) * 100)))

            # Create Frisbee
            self.frisbee = rendering.make_circle(40)
            self.circletrans = rendering.Transform(translation=(150 + 100 * (self.frisbee_states[0] % self.gridWidth),
                                                                150 + 100 * (self.frisbee_states[0] // self.gridWidth)))
            self.frisbee.add_attr(self.circletrans)

            # Create holes
            self.holes = []

            for i in range(len(self.holes_states)):
                state = self.holes_states[i]
                self.circletrans = rendering.Transform(translation=(150 + 100 * (state % self.gridWidth),
                                                                    150 + 100 * (state // self.gridWidth)))
                self.holes.append(rendering.make_circle(35))

                self.holes[i].add_attr(self.circletrans)

            # Create robot
            self.agent = rendering.make_circle(30)
            self.robotrans = rendering.Transform(translation=(150 + 100 * (self.init_state % self.gridWidth),
                                                              150 + 100 * (self.init_state // self.gridWidth)))
            self.agent.add_attr(self.robotrans)

            # Set color and add to viewer
            # lines
            for line in self.lines:
                line.set_color(24 / 255, 24 / 255, 24 / 255)
                self.viewer.add_geom(line)

            # frisbee
            self.frisbee.set_color(230 / 255, 44 / 255, 44 / 255)
            self.viewer.add_geom(self.frisbee)


            # holes
            for hole in self.holes:
                hole.set_color(54 / 255, 54 / 255, 54 / 255)
                self.viewer.add_geom(hole)

            # agent
            self.agent.set_color(118 / 255, 238 / 255, 0 / 255)
            self.viewer.add_geom(self.agent)

        if self.state is None:
            return None

        # Move the robot
        self.robotrans.set_translation(150 + 100 * (self.state % self.gridWidth),
                                       150 + 100 * (self.state // self.gridWidth))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()

class SafeGridWorld_static:

    def __init__(self, grid_size=4, width=None, height=None, init_plot=False):

        md = map_details(grid_size=(grid_size, grid_size))
        self.frisbee_state = md.frisbee_state
        self.holes_state = md.holes_state
        self.init_state = md.init_state
        # max_steps = horizon

        self.width = width
        self.height = height

        self.grid_size = grid_size
        # --- Get grid coordinates
        grid_id = 0
        self.grid_xy = {}
        for j in range(grid_size):
            for i in range(grid_size):
                self.grid_xy[grid_id] = []
                p0 = (float(i), float(j))
                self.grid_xy[grid_id].append(p0)
                p1 = (float(i), float(j)+1)
                self.grid_xy[grid_id].append(p1)
                p2 = (float(i)+1, float(j)+1)
                self.grid_xy[grid_id].append(p2)
                p3 = (float(i)+1, float(j))
                self.grid_xy[grid_id].append(p3)
                pavg = (p0[0] + 0.5, p0[1]+0.5)
                self.grid_xy[grid_id].append(pavg)
                grid_id += 1

        if init_plot:
            self.init_plot()

    def init_plot(self):

        xs = np.linspace(0, self.grid_size, self.grid_size+1)
        ys = np.linspace(0, self.grid_size, self.grid_size+1)
        # 4x4
        # xs = np.linspace(0, 4, 5)
        # ys = np.linspace(0, 4, 5)
        # 10x10
        # xs = np.linspace(0, 10, 11)
        # ys = np.linspace(0, 10, 11)

        self.plt = plt
        self.plt.figure(figsize=(self.width, self.height))
        self.plt.axis(False)
        self.ax = self.plt.gca()

        # --- grid lines
        for x in xs:
            self.plt.plot([x, x], [ys[0], ys[-1]], color='black', alpha=1, linestyle='-', linewidth=0.5)
        for y in ys:
            self.plt.plot([xs[0], xs[-1]], [y, y], color='black', alpha=1, linestyle='-', linewidth=0.5)

        # --- Color goal
        goal_id = self.frisbee_state[0]
        x, y = self.grid_xy[goal_id][0]

        self.color_grid(x, y, color="red", alpha=1)

        # --- Color initial state
        x, y = self.grid_xy[self.init_state][0]

        self.color_grid(x, y, color="green", alpha=0.8)

        # --- Color hole states
        for h in self.holes_state:
            x, y = self.grid_xy[h][0]
            self.color_grid(x, y, color="black", alpha=0.6)

    def color_grid(self, x, y, color=None, alpha=1):

        self.ax.add_patch(Rectangle((x, y), 1, 1, fill=True, color=color, alpha=alpha, linewidth=0.6))

    def color_traj(self, x_list, y_list, color=None):

        self.plt.plot(x_list, y_list, color=color, linestyle="-", linewidth=2)

    def bad_trajectories(self, traj):

        score = 0
        for t in range(traj.shape[0]):
            st = traj[t]
            if st in self.holes_state:
                score += 1
        # if score > 0:
        #     return (False, score)
        # else:
        #     return (True, score)
        return score

class map_details:

    def __init__(self, grid_size=(10, 10)):

        self.grid_size = grid_size

        # ---- Details
        if grid_size == (10, 10):
            self.frisbee_state = [97] # Goal
            self.holes_state = [50, 51, 52, 53, 54,
                           60, 61, 62, 63, 64,
                           70, 71, 72, 73, 74,
                           80, 81, 82, 83, 84,
                           90, 91, 92, 93, 94]
            self.holes_state.extend([95, 96])
            self.holes_state.extend([85, 86])
            self.holes_state.extend([75, 76])
            self.holes_state.extend([65, 66])
            self.holes_state.extend([55, 56])
            self.holes_state.extend([40, 41, 42, 43, 44, 45, 46])
            self.holes_state.extend([30, 31, 32, 33, 34, 35, 36])
            self.init_state = 0

        elif grid_size == (4, 4):
            self.frisbee_state = [15]
            self.holes_state = [8, 9, 12, 13]
            self.init_state = 0

def main():
    print("Hello World")


# =============================================================================== #

if __name__ == '__main__':
    main()
