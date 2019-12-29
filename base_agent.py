import random
import numpy as np
import pandas as pd
import os
import sys
import logging
import logging.handlers
import time
from absl import app
from absl import logging as absl_logging
from pysc2.agents import base_agent
from pysc2.lib import actions, features, units, named_array
from collections import namedtuple
import pickle
from functools import partial
from pysc2.env import sc2_env, run_loop

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from configs import COMBAT_UNIT_NAME, BUILDING_UNIT_NAME, RESEARCH_NAME

DATA_FILE = 'AI_agent_data'

GAMMA = 0.9
BATCH_SIZE = 128

LOG_EPISODE = 31
LOG_REWARD = 25
LOG_MODEL = 30
""" EPISODE WARNING REWARD INFO DEBUG """

logging.addLevelName(LOG_REWARD, "REWARD")
logging.addLevelName(LOG_EPISODE, "EPISODE")
logging.addLevelName(LOG_MODEL, "MODEL")
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

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Use cpu or gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.net(x)

    def save(self):
        torch.save(self.state_dict(), self.savepath)
        log.log(LOG_MODEL, f"Save model  \"{self.savepath}\"")

    def load(self):
        if os.path.isfile(self.savepath):
            self.load_state_dict(torch.load(self.savepath, map_location=device))
            log.log(LOG_MODEL, f"Load model  \"{self.savepath}\"")
            return True
        else:
            log.log(LOG_MODEL, f"Model  \"{self.savepath}\" not found")
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

    def get_my_army_by_pos(self, obs, pos1x=0, pos1y=0, pos2x=64, pos2y=64):
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
        army_type_list = [getattr(units.Terran, unit) for unit in COMBAT_UNIT_NAME]
        return [unit for unit in obs.observation.raw_units
                if unit.alliance == features.PlayerRelative.SELF
                    and unit.unit_type in army_type_list
                    and unit.x >= pos1x and unit.x < pos2x
                    and unit.y >= pos1y and unit.y < pos2y]

    def get_enemy_army_by_pos(self, obs, pos1x=0, pos1y=0, pos2x=64, pos2y=64):
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
        army_type_list = [getattr(units.Terran, unit) for unit in COMBAT_UNIT_NAME]
        return [unit for unit in obs.observation.raw_units
                if unit.alliance == features.PlayerRelative.ENEMY
                    and unit.unit_type in army_type_list
                    and unit.health > 0
                    and unit.x >= pos1x and unit.x < pos2x
                    and unit.y >= pos1y and unit.y < pos2y]

    def get_my_building_by_pos(self, obs, pos1x=0, pos1y=0, pos2x=64, pos2y=64):
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
        building_type_list = [getattr(units.Terran, unit) for unit in BUILDING_UNIT_NAME]
        return [unit for unit in obs.observation.raw_units
                if unit.alliance == features.PlayerRelative.SELF
                    and unit.unit_type in building_type_list
                    and unit.health > 0
                    and unit.x >= pos1x and unit.x < pos2x
                    and unit.y >= pos1y and unit.y < pos2y]

    def get_enemy_building_by_pos(self, obs, pos1x=0, pos1y=0, pos2x=4, pos2y=4):
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
        building_type_list = [getattr(units.Terran, unit) for unit in BUILDING_UNIT_NAME]
        return [unit for unit in obs.observation.raw_units
                if unit.alliance == features.PlayerRelative.ENEMY
                    and unit.unit_type in building_type_list
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

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state).view(self.state_size, -1).T
        next_state_batch = torch.cat(batch.next_state).view(self.state_size, -1).T
        action_batch = torch.cat(batch.action).unsqueeze(1)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch)
        state_action_values = state_action_values.gather(1, action_batch).squeeze()

        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        loss = F.mse_loss(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def set_DQN(self, SAVE_POLICY_NET, SAVE_TARGET_NET, SAVE_MEMORY):
        self.state_size = len(self.get_state(MYOBS))
        self.action_size = len(self.actions)
        self.policy_net = DQN(self.state_size, self.action_size, SAVE_POLICY_NET).to(device)
        self.target_net = DQN(self.state_size, self.action_size, SAVE_TARGET_NET).to(device)

        self.memory = ReplayMemory(10000)

        # if saved models exist
        if self.policy_net.load() and self.target_net.load():
            with open(SAVE_MEMORY, 'rb') as f:
                self.memory = pickle.load(f)
                log.log(LOG_MODEL, f"Load memory \"{SAVE_MEMORY}\"")
        else:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
            log.log(LOG_MODEL, f"Memory \"{SAVE_MEMORY}\" not found")

        self.optimizer = optim.RMSprop(self.policy_net.parameters())



Observation = namedtuple("Observation", ['single_select', 
                                         'multi_select', 
                                         'build_queue', 
                                         'cargo', 
                                         'production_queue', 
                                         'last_actions', 
                                         'cargo_slots_available',
                                         'home_race_requested',
                                         'away_race_requested',
                                         'map_name',
                                         'feature_screen',
                                         'feature_minimap',
                                         'action_result',
                                         'alerts',
                                         'game_loop',
                                         'score_cumulative',
                                         'score_by_category',
                                         'score_by_vital',
                                         'player',
                                         'control_groups',
                                         'raw_units',
                                         'raw_effects',
                                         'upgrades',
                                         'radar'])

observation = Observation(np.zeros((0,7)),
                          np.zeros((0,7)),
                          np.zeros((0, 7)),
                          np.zeros((0, 7)),
                          np.zeros((0, 2)),
                          [],
                          [0],
                          [1],
                          [1],
                          'Simple64',
                          named_array.NamedNumpyArray(np.zeros((11,64,64)),[['height_map', 'visibility_map', 'creep', 'camera', 'player_id', 'player_relative', 'selected', 'unit_type', 'alerts', 'pathable', 'buildable'], None, None]),
                          named_array.NamedNumpyArray(np.zeros((11,64,64)),[['height_map', 'visibility_map', 'creep', 'camera', 'player_id', 'player_relative', 'selected', 'unit_type', 'alerts', 'pathable', 'buildable'], None, None]),
                          [],
                          [],
                          [0],
                          named_array.NamedNumpyArray(np.zeros((13,)),['score', 'idle_production_time', 'idle_worker_time', 'total_value_units', 'total_value_structures', 'killed_value_units', 'killed_value_structures', 'collected_minerals', 'collected_vespene', 'collection_rate_minerals', 'collection_rate_vespene', 'spent_minerals', 'spent_vespene']),
                          named_array.NamedNumpyArray(np.zeros((11,5)),[['food_used', 'killed_minerals', 'killed_vespene', 'lost_minerals', 'lost_vespene', 'friendly_fire_minerals', 'friendly_fire_vespene', 'used_minerals', 'used_vespene', 'total_used_minerals', 'total_used_vespene'], ['none', 'army', 'economy', 'technology', 'upgrade']]),
                          named_array.NamedNumpyArray(np.zeros((3,3)),[['total_damage_dealt', 'total_damage_taken', 'total_healed'], ['life', 'shields', 'energy']]),
                          named_array.NamedNumpyArray(np.zeros((11,)), ['player_id', 'minerals', 'vespene', 'food_used', 'food_cap', 'food_army', 'food_workers', 'idle_worker_count', 'army_count', 'warp_gate_count', 'larva_count']),
                          np.zeros((10,2)),
                          named_array.NamedNumpyArray(np.zeros((55,46)), [None, ['unit_type', 'alliance', 'health', 'shield', 'energy', 'cargo_space_taken =', 'build_progress', "health_ratio", "shield_ratio", "energy_ratio", "display_type", "owner", "x", "y", "facing", "radius", "cloak", "is_selected", "is_blip", "is_powered", "mineral_contents", "vespene_contents", "cargo_space_max", "assigned_harvesters", "ideal_harvesters", "weapon_cooldown", "order_length", "order_id_0", "order_id_1", "tag", "hallucination", "buff_id_0", "buff_id_1", "addon_unit_type", "active", "is_on_screen", "order_progress_0", "order_progress_1", "order_id_2", "order_id_3", "is_in_cargo", 'buff_duration_remain', 'buff_duration_max', 'attack_upgrade_level', 'armor_upgrade_level', 'shield_upgrade_level']]),
                          [],
                          [],
                          [])

MyObs = namedtuple("MyObs", ["observation"])
MYOBS = MyObs(observation)
