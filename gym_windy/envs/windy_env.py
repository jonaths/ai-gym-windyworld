import gym
from gym import error, spaces, utils
import numpy as np
from PIL import Image as Image
import matplotlib.pyplot as plt

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

num_env = 0


def blow_wind():
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
    num_env = 0

    def __init__(self):

        self.rows = 4                                       # number of cols and rows
        self.cols = 3
        self.current_row = 0                                # current agent position
        self.current_col = 0
        self.n = self.rows * self.cols                       # total cells count
        self.observation_space = spaces.Discrete(self.n)    # 4 rows X 3 columns
        self.action_space = spaces.Discrete(4)              # up, right, down, left
        self.step_reward = -1
        self.done = False
        self.start_state = 0                                # top left corner [0,0]
        self.hole_state = 4                                 # top middle cell [0,1]
        self.finish_state = 8                               # top right corner [0,2]
        self.reset()
        self.verbose = True                                 # show the grid world or not
        WindyEnv.num_env += 1
        self.this_fig_num = WindyEnv.num_env
        if self.verbose:
            self.grid = np.zeros((self.rows, self.cols))
            self.grid[self.current_row, self.current_col] = 10
            self.fig = plt.figure(self.this_fig_num)
            plt.show(block=False)
            plt.axis('off')
            self._render()

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

        if col == 1:                                        # col 1 is the windy column
            row -= blow_wind()                              # adds a shift towards the hole

        new_state = self.coord2ind([row, col])

        reward = self._get_reward(new_state=new_state)

        self.state = new_state                              # sets states and new coordinates
        self.current_row = row
        self.current_col = col
        self._render()

        if self.state == self.finish_state:
            self.done = True
            return self.state, +5, self.done, None

        if self.state == self.hole_state:
            self.done = True
            return self.state, -5, self.done, None

        return self.state, reward, self.done, None

    def reset(self):
        self.state = self.start_state
        self.done = False
        return self.state

    def _render(self, mode='human', close=False):
        if not self.verbose:
            return
        img = np.zeros((self.rows, self.cols))              # restart matrix
        img[0, 1] = 0.5                                     # add hole
        img[self.current_row, self.current_col] = 0.2       # set agent position
        fig = plt.figure(self.this_fig_num)
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

    def _get_reward(self, new_state):

        reward = self.step_reward

        return reward