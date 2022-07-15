import gym
import gym_minigrid

import ca
import envs
# from ca import GridworldInteractionLoggerWrapper
from ca import GridworldInteractionFileLoggerWrapper
from ca import ParallelConstraintWrapper
from ca import RestQueryStateWrapper
from gym_minigrid.wrappers import FlatObsWrapper, RGBImgPartialObsWrapper, ImgObsWrapper, FullyObsWrapper

from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3 import A2C
from stable_baselines3.ppo.policies import ActorCriticPolicy
from stable_baselines3.common.evaluation import evaluate_policy
import argparse
from gym_minigrid.window import Window

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

# MiniGrid-LavaGapS5-v0


# env = ParallelConstraintWrapper(env)
# env = ca.MyFullyObsWrapper(env)

# env = FullyObsWrapper(env)
# env = FlatObsWrapper(env)
# env = ImgObsWrapper(env) # Get rid of the 'mission' fields
env = ca.MyFlatObsWrapper(env)                      # convert environment 5*5*3 grid  from tensor to [1...n] array  and append selected action.
env = GridworldInteractionFileLoggerWrapper(env)    # classify state/action to safe/unsafe category based on Done flag & reward=0 and store examples into .queries file
env = RestQueryStateWrapper(env)                    # query each state/action from CA Network
# env = RGBImgPartialObsWrapper(env) # Get pixel observations
# env = ImgObsWrapper(env) # Get rid of the 'mission' field
# obs = env.reset()
model = PPO("MlpPolicy", env, verbose=1)
# model = A2C(ActorCriticPolicy, env, verbose=1,seed=1)



# Train the agent for 10000 steps
model.learn(total_timesteps=10000) # change 1 to 10000 (prod)
# try:
#     with open('/mnt/d/BigData/MyWork/GitHub/ConstraintAcquisition/benchmarks/queries/minigrid/minigrid_' + str(
#         1) + ".queries", 'w') as f:
#         # f.write(''.join(self.logs))
#         f.write(''.join(ca.hlogs))
# except FileNotFoundError:
#         print("File not found, unable to save .queries file")
# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1000) # change 1 to 100 (prod)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
