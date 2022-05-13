from gym_minigrid.register import register

from .minigrid import *

# Minigrid environments
register(
    id='MiniGrid-Combination-Picker-8x8-v0',
    entry_point='envs:CombinationPickerEnv8x8'
)

register(
    id='MiniGrid-Combination-Picker-Random-8x8-v0',
    entry_point='envs:CombinationPickerRandomEnv8x8'
)

register(
    id='MiniGrid-Combination-Picker-16x16-v0',
    entry_point='envs:CombinationPickerEnv16x16'
)

register(
    id='MiniGrid-Combination-Picker-Random-16x16-v0',
    entry_point='envs:CombinationPickerRandomEnv16x16'
)

register(
    id='MiniGrid-Combination-Picker-32x32-v0',
    entry_point='envs:CombinationPickerEnv32x32'
)

register(
    id='MiniGrid-Combination-Picker-Random-32x32-v0',
    entry_point='envs:CombinationPickerRandomEnv32x32'
)
