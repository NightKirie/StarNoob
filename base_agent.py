import random
import numpy as np
import pandas as pd
import os
import sys
import logging
import logging.handlers
import time
from absl import app
from pysc2.agents import base_agent
from pysc2.lib import actions, features, units

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from absl import logging as absl_logging
DATA_FILE = 'AI_agent_data'

LOG_EPISODE = 31
LOG_REWARD = 25
""" EPISODE WARNING REWARD INFO DEBUG """

logging.addLevelName(LOG_REWARD, "REWARD")
logging.addLevelName(LOG_EPISODE, "EPISODE")
log = logging.getLogger(name="StarNoob")
log.addFilter(logging.Filter('StarNoob'))
log.propagate = False

log.setLevel(logging.INFO)  # global

formatter = logging.Formatter(
    fmt='%(asctime)s %(module)20s:%(lineno)-3d %(levelname)7s: %(message)s',
    datefmt='%m/%d %H:%M:%S')
log_filename = f'logs/{time.strftime("%Y_%m_%d_%H%M%S", time.localtime())}.log'

ch = logging.StreamHandler()
ch.setLevel(logging.WARNING)     # starnoob logging
ch.setFormatter(formatter)

fh = logging.handlers.RotatingFileHandler(log_filename, "w", 100000)
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)

log.addHandler(ch)
log.addHandler(fh)

# close pysc2 logging
absl_logging.set_verbosity(absl_logging.FATAL)

COMBAT_UNIT_NAME = [
    "Marine",
    "Reaper",
    "Marauder",
    "Ghost",
    "Hellion",
    "SiegeTank",
    "WidowMine",
    "Hellbat",
    "Thor",
    "Liberator",
    "Cyclone",
    "VikingFighter",
    "Medivac",
    "Raven",
    "Banshee",
    "Battlecruiser"]

BUILDING_UNIT_NAME = [
    "Armory",
    "Barracks",
    "BarracksFlying",
    "BarracksReactor",
    "BarracksTechLab",
    "Bunker",
    "CommandCenter",
    "CommandCenterFlying",
    "EngineeringBay",
    "Factory",
    "FactoryFlying",
    "FactoryReactor",
    "FactoryTechLab",
    "FusionCore",
    "GhostAcademy",
    "MissileTurret",
    "OrbitalCommand",
    "OrbitalCommandFlying",
    "PlanetaryFortress",
    "Reactor",
    "Refinery",
    "RefineryRich",
    "SensorTower",
    "Starport",
    "StarportFlying",
    "StarportReactor",
    "StarportTechLab",
    "SupplyDepot",
    "SupplyDepotLowered",
    "TechLab",
]

class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9):
        self.actions = actions
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation, e_greedy=0.9):
        """
        choose a action in action pool
        Args:  obs, e_greedy = 0.9
        return: action,  type:string
        """
        self.check_state_exist(observation)
        if np.random.uniform() < e_greedy:
            state_action = self.q_table.loc[observation, :]
            action = np.random.choice(
                state_action[state_action == np.max(state_action)].index)
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        """
        learning
        Args: previous_state(list), previous_action(string), reward(variable), this_state(list)
        Returns: none
        """
        if s == s_:
            return
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.reward_decay * self.q_table.loc[s_, :].max()
        else:
            q_target = r
        self.q_table.loc[s, a] += self.learning_rate * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions),
                                                         index=self.q_table.columns,
                                                         name=state))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, state_size, action_size, savepath):
        super(DQN, self).__init__()

        self.savepath = savepath
        self.net = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, x):
        return self.net(x)

    def save(self):
        torch.save(self.state_dict(), self.savepath)

    def load(self):
        if os.path.isfile(self.savepath):
            self.load_state_dict(torch.load(self.savepath))
            return True
        else:
            return False


class BaseAgent(base_agent.BaseAgent):
    def get_my_units_by_type(self, obs, unit_type):
        """ get all user's units of a type
        Args:
            observation
            unit_type (int)

        Returns:
            list: a list of units
        """
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.alliance == features.PlayerRelative.SELF]

    def get_enemy_units_by_type(self, obs, unit_type):
        """ get all enemy's units of a type

        Args:
            observation
            unit_type (int)

        Returns:
            list: a list of units
        """
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.alliance == features.PlayerRelative.ENEMY]

    def get_my_units_by_pos(self, obs, pos1x, pos1y, pos2x, pos2y):
        """ get user's units in a position range

        Args:
            observation
            pos1x (float): x of left-top side
            pos1y (float): y of left-top side
            pos2x (float): x of right-bottom side
            pos2y (float): y of right-bottom side

        Returns:
            list: a list of units
        """
        return [unit for unit in obs.observation.raw_units
                if unit.alliance == features.PlayerRelative.SELF
                and unit.x >= pos1x and unit.x < pos2x
                and unit.y >= pos1y and unit.y < pos2y]

    def get_enemy_units_by_pos(self, obs, pos1x, pos1y, pos2x, pos2y):
        """get enemy's units in a position range

        Args:
            observation
            pos1x (float): x of left-top side
            pos1y (float): y of left-top side
            pos2x (float): x of right-bottom side
            pos2y (float): y of right-bottom side

        Returns:
            list: a list of units
        """
        return [unit for unit in obs.observation.raw_units
                if unit.alliance == features.PlayerRelative.ENEMY
                and unit.x >= pos1x and unit.x < pos2x
                and unit.y >= pos1y and unit.y < pos2y]

    def get_my_completed_units_by_type(self, obs, unit_type):
        """ get a list of user's complete building of a type

        Args:
            observation
            unit_type (int)

        Returns:
            list: a list of units
        """
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.build_progress == 100
                and unit.alliance == features.PlayerRelative.SELF]

    def get_enemy_completed_units_by_type(self, obs, unit_type):
        """ get a list of enemy's complete building of a type

        Args:
            observation
            unit_type (int)

        Returns:
            list: a list of units
        """
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.build_progress == 100
                and unit.alliance == features.PlayerRelative.ENEMY]

    def get_my_army(self, obs, pos1x=0, pos1y=0, pos2x=64, pos2y=64):
        """ get a list of my army units in a position range

        Args:
            observation
            pos1x (float): x of left-top side
            pos1y (float): y of left-top side
            pos2x (float): x of right-bottom side
            pos2y (float): y of right-bottom side

        Returns:
            list: a list of army units
        """
        return [unit for unit in obs.observation.raw_units
                if unit.alliance == features.PlayerRelative.SELF
                    and unit.unit_type in [getattr(units.Terran, unit) for unit in COMBAT_UNIT_NAME]
                    and unit.x >= pos1x and unit.x < pos2x
                    and unit.y >= pos1y and unit.y < pos2y]

    def get_enemy_army(self, obs, pos1x=0, pos1y=0, pos2x=64, pos2y=64):
        """ get a list of my army units in a position range

        Args:
            observation
            pos1x (float): x of left-top side
            pos1y (float): y of left-top side
            pos2x (float): x of right-bottom side
            pos2y (float): y of right-bottom side

        Returns:
            list: a list of army units
        """
        return [unit for unit in obs.observation.raw_units
                if unit.alliance == features.PlayerRelative.ENEMY
                    and unit.unit_type in [getattr(units.Terran, unit) for unit in COMBAT_UNIT_NAME]
                    and unit.x >= pos1x and unit.x < pos2x
                    and unit.y >= pos1y and unit.y < pos2y]

    def get_my_building(self, obs, pos1x=0, pos1y=0, pos2x=64, pos2y=64):
        """ get a list of my building units in a position range

        Args:
            observation
            pos1x (float): x of left-top side
            pos1y (float): y of left-top side
            pos2x (float): x of right-bottom side
            pos2y (float): y of right-bottom side

        Returns:
            list: a list of building units
        """
        return [unit for unit in obs.observation.raw_units
                if unit.alliance == features.PlayerRelative.SELF
                    and unit.unit_type in [getattr(units.Terran, unit) for unit in BUILDING_UNIT_NAME]
                    and unit.x >= pos1x and unit.x < pos2x
                    and unit.y >= pos1y and unit.y < pos2y]

    def get_enemy_building(self, obs, pos1x=0, pos1y=0, pos2x=4, pos2y=4):
        """ get a list of enemy building units in a position range

        Args:
            observation
            pos1x (float): x of left-top side
            pos1y (float): y of left-top side
            pos2x (float): x of right-bottom side
            pos2y (float): y of right-bottom side

        Returns:
            list: a list of building units
        """
        return [unit for unit in obs.observation.raw_units
                if unit.alliance == features.PlayerRelative.ENEMY
                    and unit.unit_type in [getattr(units.Terran, unit) for unit in BUILDING_UNIT_NAME]
                    and unit.x >= pos1x and unit.x < pos2x
                    and unit.y >= pos1y and unit.y < pos2y]

    def do_nothing(self, obs):
        return actions.RAW_FUNCTIONS.no_op()

    def step(self, obs):
        super(BaseAgent, self).step(obs)
        if obs.first():
            command_center = self.get_my_units_by_type(
                obs, units.Terran.CommandCenter)[0]
            self.base_top_left = (command_center.x < 32)
