from gym.envs.registration import register

register(
    id='windy-v0',
    entry_point='gym_windy.envs:WindyEnv',
)

