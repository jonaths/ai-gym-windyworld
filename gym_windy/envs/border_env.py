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

def binary_blow_wind():
    """
    Generates a "wind" current in a range [0,2] with a
    fixed distribution. The probability of blowing 2
    cells is 10% or less.
    :ret
    """
    s = random.random()
    return s < 0.05


class BorderEnv(gym.Env):
    """
    A 3 rows by 4 columns gridworld
    - Start cell C
    - Goal cell G
    - Cliff cell X

    cols   0123456789 
    rows 0 OOOOOOOOOO
         1 OOOOOOOOOO
         2 OOOOOOOOOO
         3 SOOXXXXOOG

    Rewards are -1 for all O cells, 10 for G and -10 for X cell.
    Done is set when agent reaches X or E.
    """

    metadata = {'render.modes': ['human']}
    num_env = 10

    def __init__(self):

        self.rows = 4  # number of cols and rows
        self.cols = 10
        self.state_type = 'index'
        self.state = None
        self.start_state = 3
        self.hole_state = [15, 19, 23, 27]
        self.finish_state_one = 39
        self.current_row, self.current_col = self.ind2coord(self.start_state)
        self.n = self.rows * self.cols  # total cells count
        self.observation_space = spaces.Discrete(self.n)  # 4 rows X 3 columns
        self.action_space = spaces.Discrete(4)  # left, right
        self.available_actions = [0, 1, 2, 3]
        self.step_reward = -1
        self.done = False

        self.fig = None
        self.sequence = []
        self.max_steps = 30  # maximum steps number before game ends
        self.sum_reward = 0
        self.walls = []
        self.default_elevation = 5

    def set_state_type(self, state_type):
        self.state_type = state_type

    def get_state_to_return(self):
        if(self.state_type == 'coord'):
            # regresa las coordenadas como estado
            state_to_return = self.ind2coord(self.state)
        elif(self.state_type == 'onehot'):
            # regresa una codificacion onehot
            state_to_return = self.ind2onehot(self.state)
        else:
            # regresa un entero
            state_to_return = self.state
        return state_to_return

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

        new_state = self.coord2ind([row, col])

        if new_state not in self.walls:
            self.state = new_state

        self.current_row, self.current_col = self.ind2coord(self.state)

        self.sequence.append(self.state)

        reward = self._get_reward(state=new_state)

        if new_state in self.hole_state:
            self.done = True

        if len(self.sequence) >= self.max_steps:
            self.done = True  # ends if max_steps is reached

        state_to_return = self.get_state_to_return()

        return state_to_return, reward, self.done, {
            'step_seq': self.sequence,
            'sum_reward': self.sum_reward,
            'elevation': 0}

    def reset(self):
        self.state = self.start_state
        coords = self.ind2coord(self.state)
        self.current_row = coords[0]
        self.current_col = coords[1]
        self.sequence = []
        self.sum_reward = 0
        self.done = False
        state_to_return = self.get_state_to_return()
        return state_to_return

    def render(self, mode='human', close=False):

        # just to prevent issues with other figures
        fig_num = 15
        if self.fig is None:
            self.fig = plt.figure(fig_num)
            plt.show(block=False)
            plt.axis('off')

        # restart matrix
        img = np.zeros((self.rows, self.cols))

        # add exit
        i, j = self.ind2coord(self.finish_state_one)
        img[i, j] = 0.20

        # add hole
        for hole in self.hole_state:
            i, j = self.ind2coord(hole)
            img[i, j] = 0.8

        # add walls
        for s in self.walls:
            i, j = self.ind2coord(s)
            img[i, j] = 0.4

        # set agent position
        img[self.current_row, self.current_col] = 0.6

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

        # assert (index >= 0)
        # assert(index < self.n - 1)

        col = index // self.rows
        row = index % self.rows

        return [row, col]

    def ind2onehot(self, index):
        assert(index < self.rows * self.cols)
        onehot = np.zeros(self.rows * self.cols)
        onehot[index] = 1
        return onehot

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

        reward = self.step_reward

        if state in self.hole_state:
            reward -= 10
            self.done = True  # ends if max_steps is reached

        elif state == self.finish_state_one:
            self.done = True
            reward += 10

        self.sum_reward += reward

        return reward

    def get_possible_actions(self, state):
        return self.available_actions