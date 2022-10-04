import argparse
import os
import os.path as osp
import subprocess
import time

import gym
import gym_minigrid  # not used, but necessary for gym registration
import stable_baselines3.common.evaluation
from gym_minigrid.wrappers import RGBImgPartialObsWrapper

import envs  # not used, but necessary for gym registration
import gym_snake  # not used, but necessary for gym registration

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
import sb3_contrib.common.maskable.evaluation

import torch

import ca
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
parser.add_argument(
    "-b",
    "--backend",
    help="Path to java jar file for CA backend process (None: connect to running process)s",
    default=None,
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

if use_carl and args.backend is not None:
    if not (osp.exists(args.backend) or osp.isfile(args.backend)):
        raise FileNotFoundError(f"Backend file {args.backend} does not exist")

    # Cheap hack: assume we are in the repo, either in rl/ subdir or main directory
    working_dir = ".." if os.getcwd().endswith("rl") else "."

    old_style_queries_file = osp.join(working_dir, "benchmarks", "queries", "minigrid", "queries.txt")
    if osp.exists(old_style_queries_file):
        os.unlink(old_style_queries_file)

    bios.EXAMPLE_PATH = old_style_queries_file

    if args.env.lower().startswith("minigrid-combination-picker"):
        bias_name = "combinationpicker"
    else:
        bias_name = "minigrid"

    # Finally, start CA backend process
    cmd = ["java", "-jar", osp.abspath(args.backend), "-f", bias_name]
    backend_process = subprocess.Popen(cmd, cwd=working_dir)
    print("Started backend process, wait for 60 sec")
    time.sleep(60)
else:
    backend_process = None

env = gym.make(args.env)
env.seed(args.seed)
env = Monitor(env, filename=bios.GYM_MONITOR_PATH)  # from sb3 for logging

if is_minigrid_env:
    # TODO Use flat obs only for CA interaction, but use CNN+MLP for RL (as in CGRL paper)
    # env = ca.FlatObsMinigridWrapper(env)
    env = gym_minigrid.wrappers.RGBImgPartialObsWrapper(env)  # Get pixel observations
    env = gym_minigrid.wrappers.ImgObsWrapper(env)

if use_carl:
    if args.carl == "mask":
        if is_minigrid_env:
            mask_fn = ca.mask_fn_minigrid
        else:
            raise Exception("No masking function defined for environment", args.env)

        env = ActionMasker(env, mask_fn)
    elif args.carl == "replace":
        pass

    # Separate variable to avoid logging during evaluation
    env_train = ca.GridworldInteractionFileLoggerWrapper(env)
else:
    env_train = env

policy_kwargs = {} #dict(activation_fn=torch.nn.ReLU,
                    # net_arch=[128, 64, 32,dict(pi=[32, 32], vf=[32,32])])

if use_carl:
    model = MaskablePPO("CnnPolicy", env_train, n_steps=512, verbose=1, seed=args.seed, policy_kwargs=policy_kwargs)
else:
    model = PPO("CnnPolicy", env_train, n_steps=512, verbose=1, seed=args.seed, policy_kwargs=policy_kwargs)

# Train the agent for `num_steps` steps
new_logger = configure(bios.GYM_LOGGER_PATH, ["stdout", "csv", "tensorboard"])
model.set_logger(new_logger)
model.learn(total_timesteps=args.num_steps, eval_env=env, eval_freq=100_000)  # change 1 to 10000 (prod)

print("Learning complete")

# Evaluate the trained agent
if use_carl:
    mean_reward, std_reward = sb3_contrib.common.maskable.evaluation.evaluate_policy(model, env,
                                                                                     n_eval_episodes=100)
else:
    mean_reward, std_reward = stable_baselines3.common.evaluation.evaluate_policy(model, env,
                                                                                  n_eval_episodes=100)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

print("call to CA Server : ", ca.CACalls)
print("obtain from  CA Server Cache : ", ca.CACache)
print("obtain from RL Cache : ", ca.RLCache)
print("Skip false actions ", ca.CASkipAction)
print("query cache size: ", len(ca.cacheObsr))
print("CA Server cache size: ", len(ca.cacheCAserver))
print("Result mismatch bet CA server & cache .queries : ", ca.dup_diff_results)

if backend_process is not None:
    backend_process.terminate()
