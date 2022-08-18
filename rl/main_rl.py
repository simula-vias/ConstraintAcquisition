import os
import os.path as osp

import gym
import gym_minigrid
from stable_baselines3 import PPO

import envs

import ca

from stable_baselines3.common.monitor import Monitor
import argparse
import sb3_contrib as sb3

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.wrappers import ActionMasker

from stable_baselines3.common.logger import configure
import bios

# Set the environment; minigrid names are registered in envs/__init__.py
# env = gym.make('MiniGrid-Combination-Picker-8x8-v0')
# env = gym.make("MiniGrid-Empty-5x5-v0")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    help="gym environment to load",
    default=bios.GYM_ENVIRONMENT
)
parser.add_argument(
    "-n",
    "--num-steps",
    type=int,
    help="Number of steps for training",
    default=bios.STEPS
)
parser.add_argument(
    "-c",
    "--carl",
    choices=["none", "mask", "replace"],
    default=bios.CARL,
)
parser.add_argument(
    "--seed",
    type=int,
    help="random seed to generate the environment with",
    default=bios.SEED,
)
parser.add_argument(
    "--log-dir",
    help="Base directory for logging",
    default=bios.LOG_BASE_DIRECTORY
)
parser.add_argument(
    "-r",
    "--run-name",
    help="Run name for logs",
    default=bios.RUN_NAME,
)

args = parser.parse_args()

# Set filenames for log files
run_directory = osp.join(args.log_dir, args.run_name)
os.makedirs(run_directory, exist_ok=True)

# Update bios module
bios.RUN_NAME = args.run_name
bios.LOGS_PATH = osp.join(run_directory, "logs.txt")
bios.EXAMPLE_PATH = osp.join(run_directory, "queries.txt")
bios.GYM_LOGGER_PATH = osp.join(run_directory, "logger.txt")
bios.GYM_MONITOR_PATH = osp.join(run_directory, "monitor.csv")

use_carl = args.carl in ("mask", "replace")
is_minigrid_env = args.env.lower().startswith("minigrid-")

env = gym.make(args.env)
env = ca.FlatObsImageOnlyWrapper(env)

if use_carl:
    env = ca.GridworldInteractionFileLoggerWrapper(env)

    if args.carl == "mask":
        if is_minigrid_env:
            mask_fn = ca.mask_fn_minigrid
        else:
            raise Exception("No masking function defined for environment", args.env)

        env = ActionMasker(env, mask_fn)
    elif args.carl == "replace":
        pass

env = Monitor(env, filename=bios.GYM_MONITOR_PATH)  # from sb3 for logging
if use_carl:
    model = MaskablePPO("MlpPolicy", env, verbose=1)
else:
    model = PPO("MlpPolicy", env, verbose=1)

# Train the agent for `num_steps` steps
new_logger = configure(bios.GYM_LOGGER_PATH, ["stdout", "csv"])
model.set_logger(new_logger)

model.learn(total_timesteps=args.num_steps)  # change 1 to 10000 (prod)

# Evaluate the trained agent
if use_carl:
    mean_reward, std_reward = sb3.common.maskable.evaluation.evaluate_policy(model, env, n_eval_episodes=1000)  # change 1 to 100 (prod)
else:
    mean_reward, std_reward = sb3.common.evaluation.evaluate_policy(model, env,
                                                                                             n_eval_episodes=1000)  # change 1 to 100 (prod)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

print("call to CA Server : ", ca.CACalls)
print("obtain from  CA Server Cache : ", ca.CACache)
print("obtain from RL Cache : ", ca.RLCache)
print("Skip false actions ", ca.CASkipAction)
print("query cache size: ", len(ca.cacheObsr))
print("CA Server cache size: ", len(ca.cacheCAserver))
print("Result mismatch bet CA server & cache .queries : ", ca.dup_diff_results)
