import json
import os
import time
import gym

import operator
from collections import Counter
from functools import reduce

import numpy as np

import requests


from gym import ObservationWrapper

from multiprocessing import Pool
import gym_minigrid.minigrid as mg
from gym import error, spaces, utils, core

import ca

hlogs = set()
salogs = set()
cacheObsr = dict()
N = 1
cacheCAserver = dict()
server_health = None
CACalls = 0
CACache = 0
RLCache = 0
CASkipAction = 0
dup_diff_results =0
# Morena REST Wrapper
class RestQueryStateWrapper(ObservationWrapper):
    server_health = None
    GOOD = "1"
    BAD = "0"

    def __init__(self, env):
        super(RestQueryStateWrapper, self).__init__(env)
        self.server_health = True
        # self.prev_obs = None
        self.counter = Counter(["positive", "negative", "uncertain"])

    def step(self, action):
        safe_actions = []
        # var_dicts = []
        last_obser = self.prev_obs
        # print(len(last_obser))

        if (self.server_health):
            # print("last observation is : ", last_obser, "the action was :",action)
            # print("last observation size is : ", last_obser.size, "the action was :", action)
            # State_Action  = {'state':last_obser,'action':action}
            prev_stat = ''
            for s in last_obser:
                prev_stat = prev_stat + " " + str(int(s))
            databody = str(prev_stat) + " " + str(int(action))
            # databody = str.replace(databody,","," ")
            newdata = True if ( databody in cacheCAserver) else False

            if (self.server_health and newdata):
                # response = requests.post("http://192.168.1.107:7044/check/line",data=databody)
                restr = queryCAServer(self,databody)
                if (restr == ""):
                    self.server_health=False

                    # response.content.decode("utf-8")
                # print(restr)
                if restr.__contains__('NEGATIVE'):
                    print(databody)
                    print("un-safe observation is detected,action replaced with <left> action")
                    action = 0
                    if newdata:
                        cacheCAserver.add(databody,False)
                    self.counter["negative"] += 1
                elif restr.__contains__('POSITIVE'):
                    time.sleep(0.05)
                    if (not salogs.__contains__(databody)):
                        salogs.add(databody)
                        # print(databody)
                        # print("positive counter :",self.posCounter)
                        if newdata:
                            cacheCAserver.add(databody, True)
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

            # else:
                # add get data from cache.
                # count the cache counters


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

    # TODO :   add duplicate checker before store observaion
    # TODO :  add random
    # TODO :  lavagrid environment , check if it is there in stablebaseline3 / fronzen lake
    # TODO : done & negative
    # TODO : how to decide which action is better for negative response .Query from action for each state and select those safe action ( not terminated after action(not die) : safe)
    # TODO : check the future conference (Next week)
    # TODO : add cache (safe/un-safe action only) for state/action not to call API each time
    # TODO : 1. create lava gap .queries
    #  2. ask CA to learn constraints
    #  3. run RL again by asking action type ( safe/not safe) before taking decision
    #       3.1 if action was unsafe query all action and select which action is safe.
    #       3.2 if action was uncertain/safe  then continue explore

    def __init__(self, env):
        super(GridworldInteractionFileLoggerWrapper, self).__init__(env)
        LOGS = "../benchmarks/queries/minigrid.logs"
        with open(LOGS, "a") as g:
                logger = "episode_len,"+"rewards,"+"steps,"+"Refer to Obs/Action Cache,"+"refer to CA Result Cache,"+"CA Server Calls,"+"Obs/Action Cache Size,"+"obs/action class differ from CA Result,"+"Skip unsafe Actions,"+"positive queries,"+"negative queries,"+"dup. positive queries,"+"dup. negative queries,"+"RLCachse"
                g.write(logger + "\n")

        LOGFILE = "../benchmarks/queries/minigrid/minigrid.queries"
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

        # if done:
        #         print('done and reward :',reward)
        # obsImage = observation['image']
        # fobsImage = obsImage.flatten()

        # HS: This is lavagap/gridworld specific
        if action == 0:
            action = 7  # java carl not consider action=0

        observation_str = ' '.join([str(int(elem)) for elem in self.prev_obs])
        obs_action_pair = observation_str + " " + str(int(action))

        # HS: This is lavagap/gridworld specific
        is_safe = (not done) or (done and reward > 0)

        self.steps += 1

        if is_safe:
            self.posq = self.posq + 1
        else:
            self.negq = self.negq + 1
        # Send new observation/action pair to CA, if not already in cache
        if not obs_action_pair in cacheObsr:


            if is_safe:
                record = obs_action_pair + " 1"
                self.posq = self.posq + 1
                cacheObsr.update({obs_action_pair:True})
                print('a SAFE observation added , queries size :', self.posq)
            else:
                record = obs_action_pair + " 0"
                self.negq = self.negq + 1
                cacheObsr.update({obs_action_pair: False})
                print('a SAFE observation added , queries size :', self.posq)

                self.reset()
            LOGFILE = "../benchmarks/queries/minigrid/minigrid.queries"
            with open(LOGFILE, "a") as f:
                f.write(record + "\n")
        else:
            if (cacheObsr.get(obs_action_pair)!= is_safe):
                global dup_diff_results
                dup_diff_results = dup_diff_results +1
                if is_safe:
                        print("the state/Action cache was :"+str(cacheObsr.get(obs_action_pair)), "and now become SAFE")
                else:
                        print("the state/Action cache was :" + str(cacheObsr.get(obs_action_pair)), "and now become UN-SAFE")

            if is_safe:
                self.posq_dup +=1
                print('a SAFE duplicate observation visited , queries size :', self.posq_dup)
            else:
                record = obs_action_pair + " 0"
                self.negq_dup += 1
                print('a un-safe duplicate observation visited , queries size :', self.negq_dup)
                self.reset()

        LOGS = "../benchmarks/queries/minigrid.logs"
        with open(LOGS, "a") as g:
            if done:
                logger =str(self.env.step_count)+","+str(reward)+","+ str(self.steps)+","+str(ca.CACache)+","+str(len(ca.cacheCAserver))+","+str(ca.CACalls)+","+str(len(ca.cacheObsr))+","+str(ca.dup_diff_results)+","+str(ca.CASkipAction)+","+str(self.posq)+","+str(self.negq)+","+ str(self.posq_dup)+","+str(self.negq_dup)+","+str(ca.RLCache)
                g.write(logger + "\n")
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


class MyFlatObsWrapper(ObservationWrapper):
    """
    Encode mission strings using a one-hot scheme,
    and combine these with observed images into one flat array
    """

    def __init__(self, env, maxStrLen=96):
        super().__init__(env)

        # self.maxStrLen = maxStrLen
        # self.numCharCodes = 27

        # imgSpace = env.observation_space.spaces['image']
        # imgSize = reduce(operator.mul, imgSpace.shape, 1)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            # shape=(imgSize + self.numCharCodes * self.maxStrLen,),
            shape=(self.env.width * self.env.height * 3,),  # number of cells
            dtype='uint8'
        )

        self.cachedStr = None
        self.cachedArray = None

    # def step(self, action):
    #     return self.env.step(action)
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        # print('step=%s, reward=%.2f' % (self.env.step_count, reward))
        return self.observation(observation), reward, done, info

    # def step(self, action):
    #     obs, reward, done, info = self.env.step(action)
    #     print('step=%s, reward=%.2f' % (self.env.step_count, reward))
    #     # print(obs)

    def observation(self, obs):
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([
            mg.OBJECT_TO_IDX['agent'],
            mg.COLOR_TO_IDX['red'],
            env.agent_dir
        ])

        obs = full_grid.flatten()
        return obs


class ObjFlatObsWrapper(ObservationWrapper):

    def __init__(self, env, maxStrLen=96):
        super().__init__(env)

        self.observation_space = spaces.Box(
            low=0,
            high=10,
            # shape=(imgSize + self.numCharCodes * self.maxStrLen,),
            shape=(self.env.width * self.env.height,),  # number of cells
            dtype='uint8'
        )

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    def observation(self, obs):
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([
            10,
            0,
            env.agent_dir
        ])

        obs = full_grid.flatten()
        full_grid_obj = obs[::3]
        return full_grid_obj


class AgentLocObsWrapper(ObservationWrapper):

    def __init__(self, env, maxStrLen=96):
        super().__init__(env)

        self.observation_space = spaces.Box(
            low=0,
            high=10,
            # shape=(imgSize + self.numCharCodes * self.maxStrLen,),
            shape=(2,),  # number of cells
            dtype='uint8'
        )

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    def observation(self, obs):
        env = self.unwrapped
        # agent_loc = np.array(1,1)
        agent_loc = np.array([env.agent_pos[0], env.agent_pos[1]])

        obs = agent_loc.flatten()
        # full_grid_obj = obs[::3]
        return obs


class AgentLocationDirectionObsWrapper(ObservationWrapper):

    def __init__(self, env, maxStrLen=96):
        super().__init__(env)

        self.observation_space = spaces.Box(
            low=0,
            high=10,
            # shape=(imgSize + self.numCharCodes * self.maxStrLen,),
            shape=(3,),  # number of cells
            dtype='uint8'
        )

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    def observation(self, obs):
        env = self.unwrapped
        full_grid = env.grid.encode()
        # agent_loc = np.array(1,1)
        direction = env.agent_dir
        if direction == 0:  # CARL not accept 0 value
            direction = 4
        agent_loc = np.array([env.agent_pos[0], env.agent_pos[1], direction])

        obs = agent_loc.flatten()
        # full_grid_obj = obs[::3]
        return obs


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

        imgSpace = env.observation_space.spaces['image']
        imgSize = reduce(operator.mul, imgSpace.shape, 1)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(imgSize,),
            dtype='uint8'
        )

    def observation(self, obs):
        image = obs['image']
        return image.flatten()


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
    observation_str = ' '.join([str(int(elem)) for elem in obs])

    for i in range(env.unwrapped.action_space.n):
        obs_action_pair = observation_str + " " + str(int(7 if i==0 else i))
        result = True
        # observation/action pair exists cache,
        if obs_action_pair in cacheObsr:
            result = cacheObsr.get(obs_action_pair)
            action_mask[i] = result
            if (not result):
                global CASkipAction
                CASkipAction += 1
                print("un-safe Q-observation/action prevented by make action illegal from cache:",CASkipAction)

            global RLCache
            RLCache +=1

            print("existing Q-observation/action get from RL cache:", RLCache)
        else:
            if obs_action_pair in cacheCAserver:
                result = cacheCAserver.get(obs_action_pair)
                action_mask[i] = result
                global CACache
                CACache += 1

                print("existing Q-observation/action get from CA Server cache:", CACache)
            else:
                # Send new observation/action pair to CA, if not already in cache
                QResultStr = queryCAServer(obs_action_pair);
                global CACalls
                CACalls +=1
                print("new Q-observation/action called to CA Server:",CACalls)


                if QResultStr.__contains__("NEGATIVE"):
                    result=False

                if (not result):
                    # global CASkipAction
                    CASkipAction += 1
                    print("un-safe Q-observation/action prevented by make action illegal:",CASkipAction)
                #grant action per result
                action_mask[i] = result
                # insert new observation into cache
                if not QResultStr.__contains__("UNKNOWN"):
                    cacheCAserver.update({obs_action_pair:result})
    return action_mask



def queryCAServer(data_body)-> str:

        try:
                response = requests.post("http://localhost:7044/check/line",data=data_body)
                restr = response.content.decode("utf-8")
                time.sleep(0.05)
                # print(restr)
                return restr

        except requests.exceptions.HTTPError as error:
            print(error)
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
