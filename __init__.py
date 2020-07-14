from gym.envs.registration import register

register(
    id='foo-v0',    # name
    entry_point='gym_foo.envs:CustomEnv',  # env class (FooEnv)
)
register(
    id='foo-extrahard-v0',
    entry_point='gym_foo.envs:FooExtraHardEnv',
)
