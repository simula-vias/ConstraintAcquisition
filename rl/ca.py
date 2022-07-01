import json
import time
from typing import List, Dict

import operator
from functools import reduce
import numpy
import numpy as np
import sqlite3
import requests

import zmq
from gym import ObservationWrapper
from gym.spaces.space import Space
from gym_minigrid.minigrid import MiniGridEnv
from multiprocessing import Pool
from gym_minigrid.window import Window
from gym import error, spaces, utils

hlogs = set()
N = 1
# Morena REST Wrapper
class RestQueryStateWrapper(ObservationWrapper):
    server_health = None
    GOOD = "1"
    BAD = "0"
    def __init__(self, env):
        super(RestQueryStateWrapper, self).__init__(env)
        self.server_health = True
        # self.prev_obs = None


    def step(self, action):
        safe_actions = []
        # var_dicts = []
        last_obser = self.prev_obs
        #print(len(last_obser))

        if (self.server_health):
            #print("last observation is : ", last_obser, "the action was :",action)
            #print("last observation size is : ", last_obser.size, "the action was :", action)
            # State_Action  = {'state':last_obser,'action':action}
            prev_stat = ''
            for s in last_obser:
                prev_stat = prev_stat + " " + str(int(s))
            databody = str( prev_stat) + " " + str(int(action))

            # databody = str.replace(databody,","," ")



        response = None

        try:
            if (self.server_health):
                response = requests.post("http://192.168.1.107:7044/check/line",data=databody)
                restr = response.content.decode("utf-8")
                # print(restr)
                if restr.__contains__('NEGATIVE'):
                    # print("replace with safe action")
                    action = 0
                elif restr.__contains__('POSITIVE'):
                    print(databody)
                    print("action is possitive and allowed")
            time.sleep(0.1)
        except requests.exceptions.HTTPError as error:
            self.server_health = False
            print(error)
        except requests.exceptions.ConnectionError  as cerror:
            self.server_health = False
            print(cerror)
        except requests.exceptions.Timeout  as terror:
            print(terror)

        # if (response != None and response.status_code == 200):
        #     print(response.content)


        observation, reward, done, info = self.env.step(action)

        # if done:
        #     self.reset()
        self.prev_obs =  observation
        return observation, reward, done, info


    def reset(self, **kwargs):
        self.prev_obs = super(RestQueryStateWrapper, self).reset(**kwargs)
        return self.prev_obs

    def observation(self, observation):
        return observation
# Morena REST Wrapper

#Morena - Observation file writer
class GridworldInteractionFileLoggerWrapper(ObservationWrapper):
    # N = 1
    negq = 0
    posq = 0
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


    recNo = 1
    # logs = []

    def __init__(self, env):
        super(GridworldInteractionFileLoggerWrapper, self).__init__(env)
        # self.prev_obs = None

    def reset(self, **kwargs):
        self.prev_obs = super(GridworldInteractionFileLoggerWrapper, self).reset(**kwargs)
        return self.prev_obs

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        # if done:
        #         print('done and reward :',reward)
        # obsImage = observation['image']
        # fobsImage = obsImage.flatten()
        if(self.recNo < self.MaxRecordsNumber):


            observationStr = ' '.join([str(int(elem)) for elem in observation])

            record =' '

            if (not done ):
                record = observationStr + " " + str(int(action)) + " " + "1" + "\n"
                # print('Positive found :',reward)
            else:
                record = observationStr + " " + str(int(action)) + " " + "0" + "\n"

            # print('reward :',reward)
            # self.logs.append(record)
            if (not hlogs.__contains__(record)):
                self.recNo = self.recNo + 1
                hlogs.add(record)
                if (not done):
                    self.posq = self.posq + 1
                    print('a SAFE observation added , queries size :',self.posq)
                else :
                    self.negq = self.negq + 1
                    print('a un-safe observation added , queries size :',self.negq)
        else:
            with open('/mnt/d/BigData/MyWork/GitHub/ConstraintAcquisition/benchmarks/queries/minigrid/minigrid_'+ str(self.N)+".queries" , 'w') as f:
                # f.write(''.join(self.logs))
                f.write(''.join(self.hlogs))
            print("the filename is created :",'minigrid_', self.N)
            N +=1
            self.recNo = 1
            hlogs.clear()

        # print(data, res)
        self.prev_obs = observation
        return observation, reward, done, info

    def observation(self, observation):
        return observation
#Morena - Observation file writer

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
            shape=(self.env.width * self.env.height* 3,),  # number of cells
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
            10,
            0,
            env.agent_dir
        ])
        # image = obs['image']
        # mission = obs['mission']
        # print(mission)
        #
        # # Cache the last-encoded mission string
        # if mission != self.cachedStr:
        #     assert len(mission) <= self.maxStrLen, 'mission string too long ({} chars)'.format(len(mission))
        #     mission = mission.lower()
        #
        #     strArray = np.zeros(shape=(self.maxStrLen, self.numCharCodes), dtype='float32')
        #
        #     for idx, ch in enumerate(mission):
        #         if ch >= 'a' and ch <= 'z':
        #             chNo = ord(ch) - ord('a')
        #         elif ch == ' ':
        #             chNo = ord('z') - ord('a') + 1
        #         assert chNo < self.numCharCodes, '%s : %d' % (ch, chNo)
        #         strArray[idx, chNo] = 1
        #
        #     self.cachedStr = mission
        #     self.cachedArray = strArray

        # obs = np.concatenate((image.flatten(), self.cachedArray.flatten()))
        obs = full_grid.flatten()
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


# class GridworldInteractionLoggerWrapper(ObservationWrapper):
#     def __init__(self, env, db_file=":memory:"):
#         super(GridworldInteractionLoggerWrapper, self).__init__(env)
#         self.prev_obs = None
#
#         self.cell_mapping = {}
#         self.env_name = type(env.envs[0]).__name__
#         self.db = sqlite3.connect(db_file)
#         self.setup_db()
#
#     def setup_db(self):
#         cur = self.db.cursor()
#         cur.execute("CREATE TABLE IF NOT EXISTS logs (data json);")
#         cur.execute("""PRAGMA synchronous = EXTRA""")
#         cur.execute("""PRAGMA journal_mode = WAL""")
#         # cur.execute("BEGIN TRANSACTION")
#
#     def reset(self, **kwargs):
#         self.prev_obs = super(GridworldInteractionLoggerWrapper, self).reset(**kwargs)
#         return self.prev_obs
#
#     def step(self, action):
#         observation, reward, done, info = self.env.step(action)
#
#         logs = []
#         for obs, a, r, next_obs, d, i in zip(
#                 self.prev_obs, action, reward, observation, done, info
#         ):
#             obs_vars = self._build_var_vector(None, obs)
#             next_obs_vars = self._build_var_vector(None, next_obs)
#             logs.append(
#                 (
#                     json.dumps(
#                         {
#                             "variables": obs_vars,
#                             "next_variables": next_obs_vars,
#                             "observation": obs,
#                             "next_observation": next_obs,
#                             "action": int(a),
#                             "reward": r,
#                             "done": d,
#                             "info": i,
#                             "env": self.env_name,
#                         },
#                         cls=NumpyEncoder,
#                     ),
#                 )
#             )
#
#         self.db.executemany("INSERT INTO logs VALUES (?);", logs)
#         # print(data, res)
#         self.db.commit()
#         self.prev_obs = observation
#         return observation, reward, done, info
#
#     def _build_var_vector(self, action, obs):
#         return _build_var_vector(action, obs, self.cell_mapping)
#
#     def observation(self, observation):
#         return observation


# class ActionReplacementWrapper(ParallelConstraintWrapper):
#     def __init__(self, env, solver="native"):
#         super(ActionReplacementWrapper, self).__init__(env)
#
#         self.solver = solver
#
#         if solver != "native":
#             self.pool.close()
#             self.pool = None
#
#         self.default_action = MiniGridEnv.Actions.right
#         self.cell_mapping = {(-1, -1): 0}
#
#         self.state_set = set([])
#
#         # TODO Cache queries that cannot change; only CA positive case?
#         self.query_cache = {}
#
#     def step(self, action):
#         try:
#             safe_actions = []
#             var_dicts = []
#             for a, o in zip(action, self.prev_obs):
#                 new_action, var_dict = self._replace_action(a, o)
#                 safe_actions.append(new_action)
#                 var_dicts.append(var_dict)
#         except TypeError as e:
#             safe_actions, var_dicts = self._replace_action(action, self.prev_obs)
#
#         self.prev_obs, reward, done, info = self.env.step(safe_actions)
#
#         for var_dict, r in zip(var_dicts, reward):
#             var_dict["label"] = r < 0  # TODO Define safety definition function
#             self._send_training_example(var_dict)
#
#         return self.prev_obs, reward, done, info
#
#     def _replace_action(self, action, prev_obs):
#         var_dict = {"variables": self._build_var_vector(action, prev_obs)}
#         is_safe = self._query_state(var_dict)
#
#         if not is_safe:
#             return self.default_action, var_dict
#         else:
#             return action, var_dict
#
#     def _build_var_vector(self, action, obs):
#         return _build_var_vector(action, obs, self.cell_mapping)
#
#     def _send_training_example(self, var_dict):
#         pass
#
#     def _query_state(self, var_dict) -> bool:
#         pass


# class ActionReplacementWrapperMQ(ActionReplacementWrapper):
#     def __init__(self, env, solver="native"):
#         super(ActionReplacementWrapperMQ, self).__init__(env, solver)
#         # ZeroMQ
#         self.mq_context = zmq.Context()
#         self.socket = self.mq_context.socket(zmq.REQ)
#         self.socket.connect("tcp://localhost:33154")
#         self.queried_set = set()
#         self.queried_total = 0
#         self.safe_states = set()
#
#     def _send_training_example(self, var_dict):
#         msg_string = json.dumps(var_dict)
#         msg_string_hash = hash(msg_string)
#
#         if msg_string_hash not in self.state_set:
#             self.socket.send_string(msg_string)
#             self.socket.recv()  # Receive response, but ignore result
#             self.state_set.add(msg_string_hash)
#
#     def _query_state(self, var_dict) -> bool:
#         msg_string = json.dumps(var_dict)
#         # if msg_string_hash in self.safe_states:
#
#         self.socket.send_string(msg_string)
#
#         msg_string_hash = hash(msg_string)
#         self.queried_set.add(msg_string_hash)
#         self.queried_total += 1
#
#         if self.queried_total % 500 == 0:
#             print(
#                 f"{self.queried_total} queries / {len(self.queried_set) / self.queried_total} unique"
#             )
#
#         # Wait for response
#         resp = self.socket.recv()
#         is_safe = not json.loads(resp)["result"]
#         return is_safe


# class ActionReplacementWrapperDb(ActionReplacementWrapper):
#     def __init__(self, env, solver="native"):
#         super(ActionReplacementWrapperDb, self).__init__(env, solver)
#
#         # TODO Connect to database
#         raise NotImplementedError()
#
#     def _send_training_example(self, var_dict):
#         raise NotImplementedError()
#
#     def _replace_action(self, action, prev_obs):
#         example = {"variables": self._build_var_vector(action, prev_obs)}
#         self.socket.send_string(json.dumps(example))
#
#         # Wait for response
#         resp = self.socket.recv()
#         is_safe = json.loads(resp)["result"]
#
#         if not is_safe:
#             return self.default_action, example
#         else:
#             return action, example


# def _build_var_vector(action: int, obs: Dict, mapping_dict: Dict) -> List[int]:
#     variables = []
#
#     for c in obs["image"].reshape((-1, obs["image"].shape[-1])):
#         cell = tuple(c[0:2])
#
#         if cell not in mapping_dict:
#             mapping_dict[cell] = len(mapping_dict)
#
#         variables.append(mapping_dict[cell])
#
#     for it in obs["inventory"]:
#         item = tuple(it)
#
#         if item not in mapping_dict:
#             mapping_dict[item] = len(mapping_dict)
#
#         variables.append(mapping_dict[item])
#
#     variables.append(int(obs["direction"]))
#
#     if action is not None:
#         variables.append(int(action))
#
#     return variables


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
