import gym
from gym import error, spaces, utils
import numpy as np
from PIL import Image as Image
import matplotlib.pyplot as plt
import random
import sys

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

num_env = 0


def uniform_blow_wind():
    """
    Generates a "wind" current in a range [0,2] with a
    given normal distribution
    :return:
    """
    mu, sigma = 1, 0.5                                  # mean and standard deviation
    s = np.random.normal(mu, sigma, 1)[0]
    if s < 0.5:                                         # discretize distribution in 3 bins
        shift = 0
    elif 0.5 <= s < 1.5:
        shift = 1
    else:
        shift = 2
    return shift

def easy_blow_wind():
    """
    Generates a "wind" current in a range [0,2] with a
    fixed distribution. The probability of blowing 2
    cells is 10% or less.
    :ret
    """
    s = random.random()
    if s < 0.7:                                         # discretize distribution in 3 bins
        shift = 0
    elif 0.7 <= s < 0.9:
        shift = 1
    else:
        shift = 2
    return shift

def binary_blow_wind():
    """
    Generates a "wind" current in a range [0,2] with a
    fixed distribution. The probability of blowing 2
    cells is 10% or less.
    :ret
    """
    s = random.random()
    if s < 0.15:                                         # discretize distribution in 3 bins
        shift = 1
    else:
        shift = 0
    return shift

class WindyEnv(gym.Env):
    """
    A windy 3 rows by 4 columns grid world

    cols   012
    rows 0 SXE
         1 OOO
         2 OOO
         3 OOO
            ^

    Agent starts at S and the task is to reach E avoiding X.
    Column 1 is windy and pushes the agent towards X.
    The wind strength is 0, 1 or 2 cells.
    Rewards are -1 for all O cells, 5 for E and -5 for X cell.
    Done is set when agent reaches X or E.
    """

    metadata = {'render.modes': ['human']}
    num_env = 10

    def __init__(self):

        self.rows = 3                                       # number of cols and rows
        self.cols = 3
        self.current_row = 0                                # current agent position
        self.current_col = 0
        self.n = self.rows * self.cols                      # total cells count
        self.observation_space = spaces.Discrete(self.n)    # 4 rows X 3 columns
        self.action_space = spaces.Discrete(4)              # up, right, down, left
        self.step_reward = -1
        self.done = False
        self.start_state = 0                                # top left corner [0,0]
        self.hole_state = 3                                 # top middle cell [0,1]
        self.finish_state = 6                               # top right corner [0,2]
        self.fig = None
        self.sequence = []
        self.max_steps = 15                                 # maximum steps number before game ends
        self.sum_reward = 0

    def init_render(self):
        self.grid = np.zeros((self.rows, self.cols))
        self.grid[self.current_row, self.current_col] = 10
        self.fig = plt.figure(self.this_fig_num)
        plt.show(block=False)
        plt.axis('off')
        self.verbose = True  # show the grid world or not

    def step(self, action):
        assert self.action_space.contains(action)
        assert not self.done, "Already done. You should not do this. Call reset(). "

        [row, col] = self.ind2coord(self.state)

        if action == UP:                                    # validates edges
            row = max(row - 1, 0)
        elif action == DOWN:
            row = min(row + 1, self.rows - 1)
        elif action == RIGHT:
            col = min(col + 1, self.cols - 1)
        elif action == LEFT:
            col = max(col - 1, 0)

        if row == 1 and col == 1:                           # in 1,1 the wind blows
            shift = binary_blow_wind()
            row = max(row - shift, 0)                       # adds a shift towards the hole

        new_state = self.coord2ind([row, col])

        reward = self._get_reward(state=new_state)

        self.state = new_state                              # sets states and new coordinates
        self.current_row = row
        self.current_col = col

        self.sequence.append(self.state)

        if len(self.sequence) >= self.max_steps:
            self.done = True                                # ends if max_steps is reached



        return self.state, reward, self.done, {'step_seq': self.sequence, 'sum_reward': self.sum_reward}

    def reset(self):
        self.state = self.start_state
        coords = self.ind2coord(self.state)
        self.current_row = coords[0]
        self.current_col = coords[1]
        self.sequence = []
        self.sum_reward = 0
        self.done = False
        return self.state

    def render(self, mode='human', close=False):
        fig_num = 15                                        # just to prevent issues with other figures
        if self.fig is None:
            self.fig = plt.figure(fig_num)
            plt.show(block=False)
            plt.axis('off')

        img = np.zeros((self.rows, self.cols))              # restart matrix
        img[0, 1] = 0.5                                     # add hole
        img[self.current_row, self.current_col] = 0.2       # set agent position
        fig = plt.figure(fig_num)
        plt.clf()
        plt.imshow(img)
        fig.canvas.draw()
        plt.pause(0.00001)
        return

    def ind2coord(self, index):
        """
        Converts an index to coordinates
        :param index: int
        :return:
        """

        assert (index >= 0)
        # assert(index < self.n - 1)

        col = index // self.rows
        row = index % self.rows

        return [row, col]

    def coord2ind(self, coord):
        """
        Converts coordinates to index
        :param coord: [x, y]
        :return:
        """

        [row, col] = coord

        assert (row < self.rows)
        assert (col < self.cols)

        return col * self.rows + row

    def _get_reward(self, state):

        # print "finish_state:", self.finish_state
        # print "hole_state:", self.hole_state
        # print "state:", state

        reward = self.step_reward

        if state == self.finish_state:
            self.done = True
            reward += 8

        if state == self.hole_state:
            self.done = True
            reward -= 3

        self.sum_reward += reward

        return reward
