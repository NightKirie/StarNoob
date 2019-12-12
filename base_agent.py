import random
import numpy as np
import pandas as pd
import os
from absl import app
from pysc2.agents import base_agent
from pysc2.lib import actions, features, units


DATA_FILE = 'AI_agent_data'


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


class BaseAgent(base_agent.BaseAgent):

    def get_my_units_by_type(self, obs, unit_type):
        """
        get all user's units of a type
        Args: observation,  unit_type(int)
        Returns: unit(list)
        """
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.alliance == features.PlayerRelative.SELF]

    def get_enemy_units_by_type(self, obs, unit_type):
        """
        get all enemy's units of a type
        Args: observation,  unit_type(int)
        Returns: unit(list)
        """
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.alliance == features.PlayerRelative.ENEMY]

    def get_my_units_by_pos(self, obs, pos1x, pos1y, pos2x, pos2y):
        """
        get user's units in a position range
        Args: observation,  position
        Returns: unit(list)
        """
        return [unit for unit in obs.observation.raw_units
                if unit.alliance == features.PlayerRelative.SELF
                and unit.x >= pos1x and unit.x < pos2x
                and unit.y >= pos1y and unit.y < pos2y]

    def get_enemy_units_by_pos(self, obs, pos1x, pos1y, pos2x, pos2y):
        """
        get enemy's units in a position range
        Args: observation,  position
        Returns: unit(list)
        """
        return [unit for unit in obs.observation.raw_units
                if unit.alliance == features.PlayerRelative.ENEMY
                and unit.x >= pos1x and unit.x < pos2x
                and unit.y >= pos1y and unit.y < pos2y]

    def get_my_completed_units_by_type(self, obs, unit_type):
        """
        get a list of user's complete building of a type
        Args: observation,  unit_type(int)
        Returns: unit(list)
        """
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.build_progress == 100
                and unit.alliance == features.PlayerRelative.SELF]

    def get_enemy_completed_units_by_type(self, obs, unit_type):
        """
        get a list of enemy's complete building of a type
        Args: observation,  unit_type(int)
        Returns: unit(list)
        """
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.build_progress == 100
                and unit.alliance == features.PlayerRelative.ENEMY]

    def do_nothing(self, obs):
        return actions.RAW_FUNCTIONS.no_op()

    def step(self, obs):
        super(BaseAgent, self).step(obs)
        if obs.first():
            command_center = self.get_my_units_by_type(
                obs, units.Terran.CommandCenter)[0]
            self.base_top_left = (command_center.x < 32)
