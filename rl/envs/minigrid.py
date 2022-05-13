import itertools

from gym_minigrid.minigrid import *


class CombinationPickerEnv(MiniGridEnv):
    """
    Four-room minigrid with the additional objective to pick up different objects.
    """

    def __init__(
            self,
            size=8,
            agent_pos=(1, 1),
            goal_pos=None,
            n_objects=3,
            n_duplicates=2,
    ):
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos

        self.max_inventory_size = 16
        self.items = (Ball, Key)

        # Reduce objects if there are too many
        if n_objects <= size / 2 + 1:
            self.n_objects = int(n_objects)
        else:
            self.n_objects = int(size / 2)

        assert (
                self.n_objects <= self.max_inventory_size
        ), "Inventory needs to fit number of objects"

        self.n_duplicates = n_duplicates
        super().__init__(
            grid_size=size,
            max_steps=8 * size * size,
            # Set this to True for maximum speed
            see_through_walls=False,
        )

        self.action_space = spaces.Discrete(self.actions.pickup + 1)

        self.observation_space = spaces.Dict(
            {
                "image": self.observation_space.spaces["image"],
                "inventory": spaces.Box(
                    low=-1,
                    high=max(len(COLOR_NAMES), len(self.items)),
                    shape=(self.max_inventory_size, 2),
                    dtype=np.int16,
                ),
            }
        )
        self.reset()

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        room_w = width // 2
        room_h = height // 2

        # For each row of rooms
        for j in range(0, 2):

            # For each column
            for i in range(0, 2):
                xL = i * room_w
                yT = j * room_h
                xR = xL + room_w
                yB = yT + room_h

                # Bottom wall and door
                if i + 1 < 2:
                    self.grid.vert_wall(xR, yT, room_h)
                    pos = (xR, self._rand_int(yT + 1, yB))
                    self.grid.set(*pos, None)

                # Bottom wall and door
                if j + 1 < 2:
                    self.grid.horz_wall(xL, yB, room_w)
                    pos = (self._rand_int(xL + 1, xR), yB)
                    self.grid.set(*pos, None)

        # Randomize the player start position and orientation
        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)
            self.agent_dir = self._rand_int(0, 4)  # assuming random start direction
        else:
            self.place_agent()

        if self._goal_default_pos is not None:
            goal = Goal()
            self.grid.set(*self._goal_default_pos, goal)
            goal.init_pos, goal.cur_pos = self._goal_default_pos
        else:
            self.place_obj(Goal())

        # Place objects
        self.inventory = -np.ones((self.max_inventory_size, 2), dtype=np.int)
        self.objects = []

        for _, (obj_class, color) in zip(
                range(0, self.n_objects), itertools.product(self.items, COLOR_NAMES)
        ):
            for _ in range(self.n_duplicates):
                self.objects.append(obj_class(color=color))
                self.place_obj(self.objects[-1], max_tries=100)

        self.mission = "pick %d different objects and reach the goal" % self.n_objects

    def step(self, action):
        obs, reward, done, info = super().step(action)

        # Agent picked up an object
        if not done and self.carrying:
            inv_repr = [
                OBJECT_TO_IDX[self.carrying.type],
                COLOR_TO_IDX[self.carrying.color],
            ]

            if np.equal(self.inventory, inv_repr).all(axis=1).any():
                reward = -1
                done = True
                self.carrying = None
            else:
                free_slots = np.equal(self.inventory, (-1, -1)).all(axis=1)
                next_slot = np.where(free_slots)[0][0]
                self.inventory[next_slot, :] = inv_repr
                self.inventory = self.inventory[
                    np.lexsort(-self.inventory[:, ::-1].T, axis=0)
                ]
                reward = 1 / self.n_objects
                self.carrying = None

        obs = self.gen_obs()
        return obs, reward, done, info

    def gen_obs(self):
        obs = super().gen_obs()
        obs["inventory"] = self.inventory
        return obs

    def _reward(self):
        picked_objects = (
                self.n_objects - np.equal(self.inventory, (-1, -1)).all(axis=1).sum()
        )
        return 1 - 0.8 * (
                self.step_count / self.max_steps
        )  # + picked_objects/self.n_objects


class CombinationPickerEnv8x8(CombinationPickerEnv):
    def __init__(self):
        super().__init__(size=8, n_objects=4)


class CombinationPickerRandomEnv8x8(CombinationPickerEnv):
    def __init__(self):
        super().__init__(size=8, agent_start_pos=None, n_objects=4)


class CombinationPickerEnv16x16(CombinationPickerEnv):
    def __init__(self):
        super().__init__(size=16, n_objects=8)


class CombinationPickerRandomEnv16x16(CombinationPickerEnv):
    def __init__(self):
        super().__init__(size=16, agent_start_pos=None, n_objects=8)


class CombinationPickerEnv32x32(CombinationPickerEnv):
    def __init__(self):
        super().__init__(size=32, n_objects=16)


class CombinationPickerRandomEnv32x32(CombinationPickerEnv):
    def __init__(self):
        super().__init__(size=32, agent_start_pos=None, n_objects=16)
