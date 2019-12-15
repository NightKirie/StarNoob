import random
import numpy as np
import pandas as pd
import os
from absl import app
from functools import partial

from pysc2.lib import actions, features, units
from pysc2.env import sc2_env, run_loop

from base_agent import QLearningTable
import base_agent

DATA_FILE = 'Sub_training_data'
MORE_MINERALS_USED_REWARD_RATE = 0.00001
MORE_VESPENE_USED_REWARD_RATE = 0.00002
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
BARRACKS = units.Terran.Barracks
FACTORY = units.Terran.Factory
STARPORT = units.Terran.Starport


class Agent(base_agent.BaseAgent):

    actions = tuple(["do_nothing"]) + \
        tuple([f"train_{unit.lower()}" for unit in COMBAT_UNIT_NAME])

    def __init__(self):
        super(Agent, self).__init__()

        # Create action function
        for unit in COMBAT_UNIT_NAME:
            self.__setattr__(
                f"train_{unit.lower()}", partial(
                    self.train_unit, unit=unit))

    def get_distances(self, obs, units, xy):
        units_xy = [(unit.x, unit.y) for unit in units]
        return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1)

    def get_least_busy_building(self, building):
        return min(building, key=lambda building: building.order_length)

    def get_building(self, obs, unit_type):
        """ get least busy building
        Args:
          obs
          unit_type (int): from pysc2.lib.units.Terran.XXX
        Returns:
          building.tag (int): if building exist
          or
          False: if building not exist
        """
        completed_buildinges = self.get_my_completed_units_by_type(
            obs, unit_type)
        if len(completed_buildinges) > 0:
            return self.get_least_busy_building(completed_buildinges).tag
        else:
            return False

    def train_unit(self, obs, unit=None):
        building_tag = False
        if unit in ["Marine", "Reaper", "Marauder", "Ghost"]:
            building_tag = self.get_building(obs, BARRACKS)
        elif unit in ["Hellion", "SiegeTank", "WidowMine", "Hellbat", "Thor", "Liberator"]:
            building_tag = self.get_building(obs, FACTORY)
        elif unit in ["Cyclone", "Viking", "Medivac", "Raven", "Banshee", "Battlecruiser"]:
            building_tag = self.get_building(obs, STARPORT)

        if building_tag:
            return actions.RAW_FUNCTIONS.__getattr__(
                f"Train_{unit}_quick")("now", building_tag)
        else:
            return actions.RAW_FUNCTIONS.no_op()


class SubAgent_Training(Agent):

    def __init__(self):
        # print('in __init__')
        super(SubAgent_Training, self).__init__()
        self.qtable = QLearningTable(self.actions)
        if os.path.isfile(DATA_FILE + '.gz'):
            self.qtable.q_table = pd.read_pickle(
                DATA_FILE + '.gz', compression='gzip')
        self.new_game()

    def reset(self):
        # print('in reset')
        super(SubAgent_Training, self).reset()
        self.new_game()

    def new_game(self):
        # print('in new game')
        self.base_top_left = None
        self.previous_state = None
        self.previous_action = None
        self.previous_total_value_units_score = 0
        self.previous_total_value_structures_score = 0
        self.previous_total_spent_minerals = 0
        self.previous_total_spent_vespene = 0

    def get_state(self, obs):
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        idle_scvs = [scv for scv in scvs if scv.order_length == 0]
        command_centers = self.get_my_units_by_type(
            obs, units.Terran.CommandCenter)
        supply_depots = self.get_my_units_by_type(
            obs, units.Terran.SupplyDepot)
        completed_supply_depots = self.get_my_completed_units_by_type(
            obs, units.Terran.SupplyDepot)
        barrackses = self.get_my_units_by_type(obs, units.Terran.Barracks)
        completed_barrackses = self.get_my_completed_units_by_type(
            obs, units.Terran.Barracks)

        marines = self.get_my_units_by_type(obs, units.Terran.Marine)

        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)

        can_afford_marine = obs.observation.player.minerals >= 100

        return (self.base_top_left,
                len(command_centers),
                len(scvs),
                len(idle_scvs),
                len(supply_depots),
                len(completed_supply_depots),
                len(barrackses),
                len(completed_barrackses),
                len(marines),
                free_supply,
                can_afford_marine,
                )

    def step(self, obs):

        if obs.last():
            self.qtable.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')
        super(SubAgent_Training, self).step(obs)
        state = str(self.get_state(obs))
        action = self.qtable.choose_action(state)
        # print(action)

        total_value_units_score = obs.observation['score_cumulative'][3]
        total_value_structures_score = obs.observation['score_cumulative'][4]
        # print(obs.observation['score_cumulative'][11])
        total_spent_minerals = obs.observation['score_cumulative'][11]
        total_spent_vespene = obs.observation['score_cumulative'][12]

        if self.previous_action is not None:
            step_reward = 0
            if total_spent_minerals > self.previous_total_spent_minerals:
                step_reward += MORE_MINERALS_USED_REWARD_RATE * \
                    (total_spent_minerals - self.previous_total_spent_minerals)
            if total_spent_vespene > self.previous_total_spent_vespene:
                step_reward += MORE_VESPENE_USED_REWARD_RATE * \
                    (total_spent_vespene - self.previous_total_spent_vespene)

            self.qtable.learn(self.previous_state,
                              self.previous_action,
                              obs.reward + step_reward,
                              'terminal' if obs.last() else state)

        self.previous_total_value_units_score = total_value_units_score
        self.previous_total_value_structures_score = total_value_structures_score
        self.previous_total_spent_minerals = total_spent_minerals
        self.previous_total_spent_vespene = total_spent_vespene
        self.previous_state = state
        self.previous_action = action
        return getattr(self, action)(obs)

    def save_module(self):
        self.qtable.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')

    def set_top_left(self, obs):
        if obs.first():
            command_center = self.get_my_units_by_type(
                obs, units.Terran.CommandCenter)[0]
            self.base_top_left = (command_center.x < 32)