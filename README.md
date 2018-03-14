
# General description
A 3 rows by 4 modified version of Sutton's windy world implemented as an Ai Gym environment. Its purpose its to test learning strategies where there is a tradeoff between a policy which yields an expected large reward with high risk (route through row 0) or an expected smaller reward but with a lower risk (route through row 3). 

     cols   012     state_labels
     rows 0 SXE     0  4  8
          1 OOO     1  5  9
          2 OOO     2  6  10
          3 OOO     3  7  11
             ^
             |
             wind direction

 - Agent starts at S and the task is to reach E avoiding X.
 - Column 1 is windy and pushes the agent towards X. The wind strength is 0, 1 or 2 cells.
 - Rewards are -1 for all O cells, 5 for E and -5 for X cell.
 - Done is set when agent reaches X or E.

# Requirements

 - PyPlot, Gym and Numpy.

# Installation

    git clone git@github.com:jonaths/ai-gym-windyworld.git
    cd ai-gym-windyworld
    pip install -e .
# Usage

    import gym
    import gym_windy
    env = gym.make("windy-v0")
    env.reset()

    # actions:
    #   UP = 0
    #   RIGHT = 1
    #   DOWN = 2
    #   LEFT = 3
    while True:
      env.render()
      action_idx = input()
      print(env.step(action_idx))
