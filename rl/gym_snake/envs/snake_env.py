import gym
from gym import spaces
from gym.utils import seeding

from gym_snake.envs.constants import Action4, Action6, GridType
from gym_snake.envs.grid import SquareGrid, HexGrid

CELL_PIXELS = 32
mode = 'agent'

class SnakeEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'pixmap'],
        'video.frames_per_second': 10
    }

    def __init__(
        self,
        grid_type=GridType.square,
        grid_size=None,
        width=None,
        height=None,
        num_snakes=1,
        num_apples=1,
        initial_snake_size=4,
        reward_apple=1,
        reward_none=0,
        reward_collision=-1,
        reward_timeout=0,
        done_apple=False,
        always_expand=False,
        max_steps=1000,
        agent_view_size = 7, #Morena
        seed=0
    ):
        assert num_snakes >= 1
        self.seed(seed)

        if grid_size:
            assert width is None and height is None
            width = grid_size
            height = grid_size

        self.width = width
        self.height = height
        self.num_snakes = num_snakes
        self.num_apples = num_apples
        self.initial_snake_size = initial_snake_size

        self.reward_apple = reward_apple
        self.reward_none = reward_none
        self.reward_collision = reward_collision
        self.reward_timeout = reward_timeout

        self.done_apple = done_apple
        self.always_expand = always_expand
        self.agent_view_size = agent_view_size
        self.actions = []
        self.action_space = []
        self.observation_space = []
        self.previous_obs = None
        self.previous_obs_flat = None

        action_class = Action4 if grid_type == GridType.square else Action6

        for _ in range(num_snakes):
            self.action_space.append(spaces.Discrete(len(action_class)))
            self.actions.append(action_class)
            self.observation_space.append(spaces.Box(
                low=0,
                high=255,
                shape=(width, height, 3),
                dtype='uint8'
            ))

        if num_snakes == 1:
            self.actions = self.actions[0]
            self.action_space = self.action_space[0]
            self.observation_space = self.observation_space[0]

        self.step_count = 0
        self.max_steps = max_steps
        self.grid_type = grid_type
        self.grid = None
        self.grid_render = None

        # self.reset()

    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, actions):
        if self.num_snakes == 1:
            assert self.action_space.contains(actions), "%r (%s) invalid" % (actions, type(actions))
            actions = [actions]

        else:
            assert len(actions) == self.num_snakes
            for i in range(self.num_snakes):
                assert self.action_space[i].contains(actions[i]), "%r (%s) invalid" % (actions[i], type(actions[i]))

        self.step_count += 1

        #added morena to save previous obs
        self.previous_obs = self.get_obs(int(self.observation_space.shape[0]))
        self.previous_obs_flat = self.previous_obs[0].flatten()

        rewards, dones = self.grid.move(actions)


        if self.step_count >= self.max_steps:
            self.grid.all_done = True
            for i in range(self.num_snakes):
                if not dones[i]:
                    rewards[i] = self.reward_timeout
                    dones[i] = True

        obs = self.get_obs(int(self.observation_space.shape[0]))

        if self.num_snakes == 1:
            obs = obs[0]
            rewards = rewards[0]
            dones = dones[0]
        # if dones == True:
        #     self.reset()
        # print("obs share",obs.shape)
        return obs, rewards, dones, {}

    def reset(self):
        self.step_count = 0
        if self.grid_type == GridType.square:
            self.grid = SquareGrid(
                np_random=self.np_random,
                width=self.width,
                height=self.height,
                num_snakes=self.num_snakes,
                num_apples=self.num_apples,
                initial_snake_size=self.initial_snake_size,
                reward_apple=self.reward_apple,
                reward_none=self.reward_none,
                reward_collision=self.reward_collision,
                done_apple=self.done_apple,
                always_expand=self.always_expand
            )
        elif self.grid_type == GridType.hex:
            self.grid = HexGrid(
                np_random=self.np_random,
                width=self.width,
                height=self.height,
                num_snakes=self.num_snakes,
                num_apples=self.num_apples,
                initial_snake_size=self.initial_snake_size,
                reward_apple=self.reward_apple,
                reward_none=self.reward_none,
                reward_collision=self.reward_collision,
                done_apple=self.done_apple,
                always_expand=self.always_expand
            )
        else:
            raise ValueError("Unrecognized grid type: ", self.grid_type)

        obs = self.get_obs(int(self.observation_space.shape[0]))

        if self.num_snakes == 1:
            obs = obs[0]

        return obs


    def get_obs(self,grid_size):
        return self.grid.encode(grid_size)

    def render(self, mode='human', close=False):
        if close:
            self.close()
            return

        if self.grid_render is None or self.grid_render.window is None:
            from gym_snake.rendering import Renderer

            if mode == 'agent':
                r_width, r_height = self.agent_view_size * CELL_PIXELS, self.agent_view_size * CELL_PIXELS
                # self.width = 7
                # self.height = 7
                self.grid.mode(mode)
                # self.grid.set_mode(mode)
            else:
                r_width, r_height = self.grid.get_renderer_dimensions(CELL_PIXELS)

            self.grid_render = Renderer(
                    r_width,
                    r_height,
                    True if (mode == 'human' or mode == 'agent') else False
            )

        r = self.grid_render

        r.beginFrame()
        self.grid.render(r, CELL_PIXELS, 4 * CELL_PIXELS)
        r.endFrame()

        if mode == 'rgb_array':
                return r.getArray()
        elif mode == 'pixmap':
            return r.getPixmap()

        return r

    def close(self):
        if self.grid_render:
            self.grid_render.close()
            self.grid_render = None
