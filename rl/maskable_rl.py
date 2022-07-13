import gym
import gym_minigrid

import ca
import envs
# from ca import GridworldInteractionLoggerWrapper
from ca import GridworldInteractionFileLoggerWrapper
from ca import ParallelConstraintWrapper
from ca import RestQueryStateWrapper
from gym_minigrid.wrappers import FlatObsWrapper, RGBImgPartialObsWrapper, ImgObsWrapper, FullyObsWrapper, SymbolicObsWrapper

from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3.ppo.policies import ActorCriticPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
import argparse
from gym_minigrid.window import Window

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.wrappers import ActionMasker

import numpy as np

# Set the environment; minigrid names are registered in envs/__init__.py
# env = gym.make('MiniGrid-Combination-Picker-8x8-v0')
# env = gym.make("MiniGrid-Empty-5x5-v0")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    help="gym environment to load",
    default='MiniGrid-LavaGapS5-v0'
)
parser.add_argument(
    "--seed",
    type=int,
    help="random seed to generate the environment with",
    default=20
)

args = parser.parse_args()

env = gym.make(args.env)

# This is the action masking function; it needs to query CA on which actions are safe/unsafe
# Potential problem: We need to figure out the current observation, because it is not a parameter of the function.
def mask_fn_lavagrid(env: gym.Env) -> np.ndarray:
    obs = env.unwrapped.gen_obs()
    forward_cell = obs["image"][7//2, 7-2]
    action_mask = np.ones(env.unwrapped.action_space.n, dtype=bool)

    if np.all(forward_cell == [9, 0, 0]):
        action_mask[2] = False

    return action_mask

# MiniGrid-LavaGapS5-v0

# Flattens the image observation and removes the mission field (we don't care about it)
env = ca.FlatObsImageOnlyWrapper(env)

# This is the new wrapper for action masking
env = ActionMasker(env, mask_fn_lavagrid)

env = Monitor(env)  # from sb3 for logging

# front_pos = [0,1,2] + env.unwrapped.gr
# 78,79,80
# agent_pos = 81,82,83
# Lava: 9,0,0

obs = env.reset()
# This is the PPO version that allows action masks; without ActionMasker it behaves the same as the normal PPO
model = MaskablePPO("MlpPolicy", env, verbose=1)

# Train the agent for 10000 steps
model.learn(total_timesteps=30000)  # change 1 to 10000 (prod)
# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1000) # change 1 to 100 (prod)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
