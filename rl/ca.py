import json
import os
import time
import gym

import operator
from collections import Counter
from functools import reduce
from datetime import datetime
import numpy as np

import requests

from gym import ObservationWrapper

from multiprocessing import Pool
from gym import spaces, core

import bios as BIOS

hlogs = set()
salogs = set()
cacheObsr = dict()
N = 1
cacheCAserver = dict()
server_health = None
CACalls = 0
CACache = 0
RLCache = 0
OASkipAction = 0
CASkipAction = 0
dup_diff_results = 0
min_ca_time = 1000
max_ca_time = 0
sumTime = 0
Ca_100_Calls = 0

# This is the action masking function; it needs to query CA on which actions are safe/unsafe
# Potential problem: We need to figure out the current observation, because it is not a parameter of the function.
def mask_fn_minigrid(env: gym.Env) -> np.ndarray:
    if env.spec.entry_point.startswith('gym_snake'):
        # obs = env.unwrapped.get_obs(env.unwrapped.height)
        # forward_cell = obs[0][env.unwrapped.width // 2, env.unwrapped.height - 2]
        # observation = obs[0]
        observation = env.unwrapped.grid.encode(int(env.observation_space.shape[0]))
        action_mask = gen_safe_actions(observation[0].flatten(), env)
    else:
        obs = env.unwrapped.gen_obs()
        forward_cell = obs["image"][7//2, 7-2]
        observation = obs['image']
    # forward_cell = obs["image"][env.front_pos[0], env.front_pos[1]]  # get front position from environment
    # action_mask = np.ones(env.unwrapped.action_space.n, dtype=bool)
    # observation = obs['image']
        action_mask = gen_safe_actions(observation.flatten(), env)
        if np.all(forward_cell == [9, 0, 0]):
        #     action_mask[2] = False
            SKIPLOGS = BIOS.SKIP_LOGS_PATH
            action_mask_str = ','.join([str(int(elem)) for elem in action_mask])
            with open(SKIPLOGS, "a") as g:
                    dt = datetime.now()
                    logger = "9," + action_mask_str +"," +str(int(datetime.timestamp(dt)*1000))
                    g.write(logger + "\n")
    return action_mask


# Morena REST Wrapper
class RestQueryStateWrapper(ObservationWrapper):
    server_health = None
    GOOD = "1"
    BAD = "0"

    def __init__(self, env):
        super(RestQueryStateWrapper, self).__init__(env)
        self.prev_obs = None
        self.server_health = True
        self.counter = Counter(["positive", "negative", "uncertain"])

    def step(self, action):
        last_obser = self.prev_obs

        if self.server_health:
            # print("last observation is : ", last_obser, "the action was :",action)
            # print("last observation size is : ", last_obser.size, "the action was :", action)
            # State_Action  = {'state':last_obser,'action':action}
            prev_stat = " ".join([str(int(s)) for s in last_obser])
            databody = str(prev_stat) + " " + str(int(action))
            # databody = str.replace(databody,","," ")
            newdata = databody not in cacheCAserver

            if newdata:
                # response = requests.post("http://192.168.1.107:7044/check/line",data=databody)
                restr = self.queryCAServer(databody)
                if restr == "":
                    self.server_health = False

                    # response.content.decode("utf-8")
                # print(restr)
                if restr.__contains__('NEGATIVE'):
                    print(databody)
                    print("un-safe observation is detected")
                    action = 0
                    if newdata:
                        cacheCAserver[databody] = False
                    self.counter["negative"] += 1
                elif restr.__contains__('POSITIVE'):
                    time.sleep(0.05)
                    if databody not in salogs:
                        salogs.add(databody)
                        # print(databody)
                        # print("positive counter :",self.posCounter)
                        if newdata:
                            cacheCAserver[databody] = True
                        self.counter["positive"] += 1
                    # print("action is possitive and allowed")
                elif restr == "":
                    pass
                else:
                    # print(databody)
                    print("uncertain observation is detected")
                    # if newdata:
                    #     cacheCAserver.add(databody, True)
                    self.counter["uncertain"] += 1

        # if (response != None and response.status_code == 200):
        #     print(response.content)

        observation, reward, done, info = self.env.step(action)

        # if done:
        #     self.reset()
        self.prev_obs = observation
        return observation, reward, done, info

    def reset(self, **kwargs):
        self.prev_obs = super(RestQueryStateWrapper, self).reset(**kwargs)
        return self.prev_obs

    def observation(self, observation):
        return observation


# Morena REST Wrapper

# Morena - Observation file writer
class GridworldInteractionFileLoggerWrapper(ObservationWrapper):
    # N = 1
    negq = 0
    posq = 0
    negq_dup = 0
    posq_dup = 0
    steps = 0
    MaxRecordsNumber = 1000

    # TODO : add duplicate checker before store observaion
    # TODO : add random
    # TODO : lavagrid environment , check if it is there in stablebaseline3 / fronzen lake
    # TODO : done & negative
    # TODO : how to decide which action is better for negative response .Query from action for each state and select those safe action ( not terminated after action(not die) : safe)
    # TODO : check the future conference (Next week)
    # TODO : add cache (safe/un-safe action only) for state/action not to call API each time
    # TODO : 1. create lava gap .queries
    #  2. ask CA to learn
    #  3. run RL again by asking action type ( safe/not safe) before taking decision
    #       3.1 if action was unsafe query all action and select which action is safe.
    #       3.2 if action was uncertain/safe  then continue explore

    def __init__(self, env):
        super(GridworldInteractionFileLoggerWrapper, self).__init__(env)

        LOGS = BIOS.LOGS_PATH
        with open(LOGS, "a") as g:
            logger = "rewards," + "episode_len," + "steps," + "Refer to Obs/Action Cache," + "refer to CA Result Cache," + "CA Server Calls," + "Obs/Action Cache Size," + "obs/action Cache CA Size," + "Skip by CA," + "positive queries," + "negative queries," + "dup. positive queries," + "dup. negative queries," + "Skip By ObsA," + "timestamp"
            g.write(logger + "\n")

        LOGFILE = BIOS.EXAMPLE_PATH
        try:
            if os.path.exists(LOGFILE):
                os.remove(LOGFILE)
        except FileNotFoundError:
            print("not able to delete old log .queries file")
        except PermissionError:
            print("not have permission to delete the old .queries file")

        # for reading existing file
        # try: # TODO : delete query file....
        #     with open(LOGFILE, "r") as f:
        #         records = f.readlines()
        #         for record in records:
        #             obs_action_pair = record[:len(record)-3]
        #             if obs_action_pair not in cacheObsr:
        #                 cacheObsr.add(obs_action_pair)
        #     print("The existing observation records cached :", len(cacheObsr))
        # except FileNotFoundError:
        #     print("no initial observation records detected")
        # for reading existing file
        self.prev_obs = None

    def reset(self, **kwargs):
        # self.env.seed = 20  # to make just on seed
        self.prev_obs = super(GridworldInteractionFileLoggerWrapper, self).reset(**kwargs)
        return self.prev_obs

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        # if reward ==0 and self.prev_obs[92] == 50 and action == 0:
        #     print('done and reward :',reward)
        #     print('done and action :', action)
        #     print('done and alldone? :', self.grid.all_done)
        #     print('done and observation :', observation)

        # obsImage = observation['image']
        # fobsImage = obsImage.flatten()

        # HS: This is lavagap/gridworld specific
        # if action == 0:
        #     action = 7  # java carl not consider action=0
        if self.env.spec.entry_point.startswith('gym_snake'):
            one_dime = squeez(self.previous_obs_flat)
            observation_str = ' '.join([str(int(elem)) for elem in one_dime])
        else:
            observation_str = ' '.join([str(int(elem)) for elem in self.prev_obs])

        obs_action_pair = observation_str + " " + str(int(action))

        # Morena : minigrid environment
        # HS: This is lavagap/gridworld specific
        is_safe = (not done) or (done and reward > 0)
        # Morena : minigrid environment

        #Morena : snake environment
        # is_safe = (reward > -1) or (not done)
        # Morena : snake environment

        # Morena : minigrid environment
        if self.env.steps_remaining == 0:
            is_safe = None
            done = True
        # self.steps += 1
        # Morena : minigrid environment

        # Send new observation/action pair to CA, if not already in cache



        #Morena minigird env
        if is_safe is not None:
            if obs_action_pair not in cacheObsr:
                if is_safe:
                    record = obs_action_pair + " 1"
                    self.posq = self.posq + 1
                    cacheObsr.update({obs_action_pair:True})
                    # print('a SAFE observation added , queries size :', self.posq)
                else:
                    record = obs_action_pair + " 0"
                    self.negq = self.negq + 1
                    cacheObsr.update({obs_action_pair: False})
                    # print('a SAFE observation added , queries size :', self.posq)

                LOGFILE = BIOS.EXAMPLE_PATH
                with open(LOGFILE, "a") as f:
                    f.write(record + "\n")
            else:
                # if (cacheObsr.get(obs_action_pair)!= is_safe):
                #     global dup_diff_results
                #     dup_diff_results = dup_diff_results + 1
                #
                #     if is_safe is None:
                #         pass
                #     elif is_safe:
                #             print("the state/Action cache was :"+str(cacheObsr.get(obs_action_pair)), "and now become SAFE")
                #             # cacheObsr.update({obs_action_pair: True})
                #             # print(str(self.prev_obs)+" "+str(action))
                #     else:
                #             print("the state/Action cache was :" + str(cacheObsr.get(obs_action_pair)), "and now become UN-SAFE")
                #             # cacheObsr.update({obs_action_pair: False})
                #             # print(str(self.prev_obs)+" "+str(action))

                if is_safe:
                    self.posq_dup += 1
                    # print('a SAFE duplicate observation visited , queries size :', self.posq_dup)
                else:
                    record = obs_action_pair + " 0"
                    self.negq_dup += 1
                    # print('a un-safe duplicate observation visited , queries size :', self.negq_dup)

        LOGS = BIOS.LOGS_PATH
        with open(LOGS, "a") as g:
            if done:
                dt = datetime.now()
                logger = str(reward) + "," + str(self.step_count) + "," + str(self.steps) + "," + str(
                    RLCache) + "," + str(CACache) + "," + str(CACalls) + "," + str(
                    len(cacheObsr)) + "," + str(len(cacheCAserver)) + "," + str(CASkipAction) + "," + str(
                    self.posq) + "," + str(self.negq) + "," + str(self.posq_dup) + "," + str(self.negq_dup) + "," + str(OASkipAction) + ","+str(int(datetime.timestamp(dt)*1000))
                g.write(logger + "\n")
                self.reset()
        if not done:
            self.prev_obs = observation
        return observation, reward, done, info

    def observation(self, observation):
        return observation


# Morena - Observation file writer
class MyFullyObsWrapper(ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding
    """

    def __init__(self, env):
        super().__init__(env)

        self.observation_space.spaces["image"] = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width, self.env.height, 3),  # number of cells
            dtype='uint8'
        )

    def observation(self, obs):
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([
            10,
            0,
            env.agent_dir
        ])

        return {
            'mission': obs['mission'],
            'image': full_grid
        }


class ParallelConstraintWrapper(ObservationWrapper):
    def __init__(self, env):
        super(ParallelConstraintWrapper, self).__init__(env)
        self.prev_obs = None

        try:
            self.workers = len(env)
            self.pool = Pool(self.workers // 2)
        except TypeError:
            self.workers = 1
            self.pool = None
        self.pool = None

    def reset(self, **kwargs):
        self.prev_obs = super(ParallelConstraintWrapper, self).reset(**kwargs)
        return self.prev_obs

    def observation(self, observation):
        return observation


class FlatObsImageOnlyWrapper(ObservationWrapper):
    """
    Encode mission strings using a one-hot scheme,
    and combine these with observed images into one flat array
    """

    def __init__(self, env):
        super().__init__(env)
        if self.spec.entry_point.startswith('gym_snake'):
            imgSpace = env.observation_space
        else:
            imgSpace = env.observation_space.spaces['image']
        imgSize = reduce(operator.mul, imgSpace.shape, 1)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(imgSize,),
            dtype='uint8'
        )

    def observation(self, obs):
        if self.spec.entry_point.startswith('gym_snake'):
            image = obs
            # one_dim_obs = np.zeros((self.env.width,self.env.height,1))
            # for width in range(self.env.width):
            #     for height in range(self.env.height):
            #         one_dim_obs[width][height][0] = obs[width][height][0]
            # img = one_dim_obs.flatten()
        else:
            img = obs['image'].flatten()
        return img


class LavaAvoidanceWrapper(core.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_obs = None

    def step(self, action):
        env = self.unwrapped

        forward_pos = np.array([0, 1, 2]) + (
                env.agent_view_size // 2 * env.agent_view_size * 3 + (env.agent_view_size - 2) * 3)
        forward_cell = self.prev_obs[forward_pos]

        if np.all(forward_cell == [9, 0, 0]) and action == 2:
            action = 1  # turn right instead

        obs, reward, done, info = self.env.step(action)

        return obs, reward, done, info

    def reset(self, **kwargs):
        self.prev_obs = self.env.reset(**kwargs)
        return self.prev_obs


def gen_safe_actions(obs, env: gym.Env) -> np.ndarray:
    action_mask = np.ones(env.unwrapped.action_space.n, dtype=bool)
    #Reduce dimension
    if env.spec.entry_point.startswith('gym_snake'):
        one_dime = squeez(obs)
        observation_str = ' '.join([str(int(elem)) for elem in one_dime])
    else:
        observation_str = ' '.join([str(int(elem)) for elem in obs])
    #Reduce dimension
    # observation_str = squeez(observation_str)
    for i in range(env.unwrapped.action_space.n):
        # CA dont accept 0 value, GYM action 0 replace with 7
        # obs_action_pair = observation_str + " " + str(int(7 if i == 0 else i))  # not needed for contextual conacq
        obs_action_pair = f"{observation_str} {i} {-1}"

        result = True
        # observation/action pair exists cache,
        # if obs_action_pair in cacheObsr:
        #     result = cacheObsr.get(obs_action_pair)
        #     action_mask[i] = result
        #     if (not result):
        #         global OASkipAction
        #         OASkipAction += 1
        #         print("un-safe Q-observation/action prevented by make action illegal from RL Observerd cache:",OASkipAction)
        #     # print("existing Q-observation/action get from RL cache:", RLCache)
        # else:
            # startTime = time.time()
            # Send new observation/action pair to CA, if not already in cache
        QResultStr = queryCAServer(obs_action_pair)

        # QResultStr = "UNKNOWN"

            # executionTime = (time.time() - startTime)
            # global max_ca_time
            # if ( executionTime> max_ca_time ):
            #     max_ca_time = executionTime
            #
            # global min_ca_time
            # if (executionTime < min_ca_time):
            #     min_ca_time = executionTime


        global CACalls
        CACalls += 1

            # global Ca_100_Calls
            # Ca_100_Calls += 1
            #
            # # global sumTime
            # # sumTime = sumTime + executionTime
            #
            # if ((Ca_100_Calls % 100)==0 ):
            #     print('Average resp.time for lass 100 calls is ',str((sumTime/Ca_100_Calls)),'(s) Min is',min_ca_time,' Max is ',max_ca_time," and total calls:",CACalls)
            #     Ca_100_Calls = 0
            #     sumTime = 0
            # # print("new Q-observation/action called to CA Server:",CACalls)

        if QResultStr.__contains__("NEGATIVE"):
            result = False


        if (not result):
            global CASkipAction
            CASkipAction += 1
            # print("un-safe Q-observation/action prevented by make action illegal:",CASkipAction)
            # grant action per result
            action_mask[i] = result
        # insert new observation into cache
        # if not QResultStr.__contains__("UNKNOWN"):
        #     cacheCAserver.update({obs_action_pair: result})
        #     cacheObsr[obs_action_pair] = result

    return action_mask


def queryCAServer(data_body) -> str:
    try:

        response = requests.post("http://" + str(BIOS.CAServerIPAddress) + ":" + str(BIOS.CAServerPort) + "/check/line",
                                 data=data_body)
        restr = response.content.decode("utf-8")
        time.sleep(float(BIOS.CAServerInterval))
        # print(restr)
        return restr

    except requests.exceptions.HTTPError as herror:
        print(herror)
    except requests.exceptions.ConnectionError as cerror:
        print(cerror)
    except requests.exceptions.Timeout as terror:
        print(terror)

    return "UNKNOWN"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
#
def squeez(three_d):
    one_d_array = np.zeros(64)
    count = 0
    leng = len(three_d)
    for i in range(0,leng,3):
        one_d_array[count] = three_d[i if i==0 else i-1]
        count = count +1
    return one_d_array

def expand(one_d):
    three_d_array = np.zeros(192)

    leng = len(one_d)
    for i in range(0,leng):
        for j in (0,3):
            three_d_array[j] = one_d[i]
    return three_d_array
