import gym
import gym_minigrid
import envs
from ca import GridworldInteractionLoggerWrapper
from ca import GridworldInteractionFileLoggerWrapper
from ca import ParallelConstraintWrapper
from ca import RestQueryStateWrapper
from gym_minigrid.wrappers import FlatObsWrapper

from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy

# Set the environment; minigrid names are registered in envs/__init__.py
# env = gym.make('MiniGrid-Combination-Picker-8x8-v0')
env = gym.make("MiniGrid-Empty-5x5-v0")
# env = FlatObsWrapper(env)
# env = GridworldInteractionLoggerWrapper(env)

env = FlatObsWrapper(env)
env = ParallelConstraintWrapper(env)
env = GridworldInteractionFileLoggerWrapper(env)
env = RestQueryStateWrapper(env)
model = PPO(MlpPolicy, env, verbose=0)

# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes= 100) # change 1 to 100 (prod)

# print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

# Train the agent for 10000 steps
model.learn(total_timesteps=10000) # change 1 to 10000 (prod)

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100) # change 1 to 100 (prod)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
