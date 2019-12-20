import random
import numpy as np
import pandas as pd
import os
from absl import app
from functools import partial
from types import SimpleNamespace
import pickle

from pysc2.lib import actions, features, units
from pysc2.env import sc2_env, run_loop

from base_agent import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from collections import namedtuple
import unit.terran_unit as terran

DATA_FILE = 'Sub_training_data'

FAILED_COMMAND = 0.00001
MORE_MINERALS_USED_REWARD_RATE = 0.00001
MORE_VESPENE_USED_REWARD_RATE = 0.00002

SAVE_POLICY_NET = 'model/training_dqn_policy'
SAVE_TARGET_NET = 'model/training_dqn_target'
SAVE_MEMORY = 'model/training_memory'

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

TRAINABLE_BUILDING = ["Barracks", "Factory", "Starport"]
BARRACKS = units.Terran.Barracks
FACTORY = units.Terran.Factory
STARPORT = units.Terran.Starport

BATCH_SIZE = 128
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 500

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class Agent(BaseAgent):

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
        elif unit in ["Cyclone", "VikingFighter", "Medivac", "Raven", "Banshee", "Battlecruiser"]:
            building_tag = self.get_building(obs, STARPORT)

        if building_tag:
            return actions.RAW_FUNCTIONS.__getattr__(
                f"Train_{unit}_quick")("now", building_tag)
        else:
            return actions.RAW_FUNCTIONS.no_op()


class SubAgent_Training(Agent):

    def __init__(self):
        super(SubAgent_Training, self).__init__()
        self.state_size = int()             # get at step
        self.action_size = len(self.actions)
        self.policy_net = nn.Module()       # DQN init at step
        self.target_net = nn.Module()       # DQN init at step

        self.optimizer = None               # init at step
        self.memory = ReplayMemory(10000)
        self.episode = 0
        self.new_game()


    def reset(self):
        log.debug('in reset')
        super(SubAgent_Training, self).reset()
        self.new_game()

    def new_game(self):
        log.debug('in new game')
        self.base_top_left = None
        self.previous_state = None
        self.previous_action = None
        self.previous_total_value_units_score = 0
        self.previous_total_value_structures_score = 0
        self.previous_total_spent_minerals = 0
        self.previous_total_spent_vespene = 0
        self.now_reward = 0

    def get_state(self, obs):
        complete_trainable_building = [len(self.get_my_completed_units_by_type(obs, getattr(terran, building)().index)) for building in TRAINABLE_BUILDING]
        complete_unit = [len(self.get_my_units_by_type(obs, getattr(terran, unit)().index)) for unit in COMBAT_UNIT_NAME]
        completed_supply_depots = self.get_my_completed_units_by_type(
            obs, terran.SupplyDepot().index)

        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)

        return tuple([self.base_top_left,
                    len(completed_supply_depots),
                    free_supply] +
                    complete_trainable_building +
                    complete_unit)


    def can_afford_unit(self, obs, unit):
        can_afford = False
        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)
        if isinstance(unit, terran.TerranCreature):
            if free_supply > 0 and \
               obs.observation.player.minerals >= unit.mineral_price and \
               obs.observation.player.vespene >= unit.vespene_price and \
               0 not in [len(self.get_my_completed_units_by_type(obs, getattr(terran, build_from)().index)) for build_from in unit.build_from] and \
               0 not in [len(self.get_my_completed_units_by_type(obs, getattr(terran, requirements)().index)) for requirements in unit.requirements]:      
                can_afford = True
        return can_afford


    def step(self, obs):
        super(SubAgent_Training, self).step(obs)

        self.episode += 1
        state = self.get_state(obs)
        log.debug(state)
        action = self.select_action(state)
        log.info(action)

        if self.previous_action is not None:
            step_reward = self.get_reward(obs)
            log.log(LOG_REWARD, "training reward = " + str(obs.reward + step_reward))
            if not obs.last:
                self.memory.push(self.previous_state,
                                 self.previous_action,
                                 state,
                                 obs.reward + step_reward)

                self.optimize_model()
            else:
                pass

        else:
            # initializaion (only execute once)
            self.state_size = len(state)
            self.policy_net = DQN(self.state_size, self.action_size, SAVE_POLICY_NET)
            self.target_net = DQN(self.state_size, self.action_size, SAVE_TARGET_NET)

            # if saved models exist
            if self.policy_net.load() and self.target_net.load():
                with open(SAVE_MEMORY, 'rb') as f:
                    self.memory = pickle.load(f)
            else:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                self.target_net.eval()

            self.optimizer = optim.RMSprop(self.policy_net.parameters())

        if self.episode % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


        self.previous_state = state
        self.previous_action = action
        return getattr(self, action)(obs)

    def get_reward(self, obs):
        total_value_units_score = obs.observation.score_cumulative.total_value_units
        total_value_structures_score = obs.observation.score_cumulative.total_value_structures
        # print(obs.observation['score_cumulative'][11])
        total_spent_minerals = obs.observation.score_cumulative.spent_minerals 
        total_spent_vespene = obs.observation.score_cumulative.spent_vespene  

        prev_reward = 0
        ## Prev reward will update in this epoch
        # If use mineral, get positive reward
        if total_spent_minerals > self.previous_total_spent_minerals:
            prev_reward += MORE_MINERALS_USED_REWARD_RATE * \
                (total_spent_minerals - self.previous_total_spent_minerals)
        # If use vespene, get positive reward
        if total_spent_vespene > self.previous_total_spent_vespene:
            prev_reward += MORE_VESPENE_USED_REWARD_RATE * \
                (total_spent_vespene - self.previous_total_spent_vespene)
        
        step_reward = prev_reward - self.now_reward

        ## Now reward will update in next epoch
        # If trying to train a unit, but there's no building to train it, get negative reward
        if False in [self.can_afford_unit(obs, getattr(terran, unit)()) for unit in COMBAT_UNIT_NAME]:
            self.now_reward = FAILED_COMMAND
        else:
            self.now_reward = 0


        self.previous_total_value_units_score = total_value_units_score
        self.previous_total_value_structures_score = total_value_structures_score
        self.previous_total_spent_minerals = total_spent_minerals
        self.previous_total_spent_vespene = total_spent_vespene
        return step_reward


    def set_top_left(self, obs):
        if obs.first():
            command_center = self.get_my_units_by_type(
                obs, units.Terran.CommandCenter)[0]
            self.base_top_left = (command_center.x < 32)

    def select_action(self, state):
        sample = random.random()
        eps_threshold = 0.9
        if sample > eps_threshold:
            with torch.no_grad():
                _, idx = self.policy_net(torch.Tensor(state)).max(0)
                return self.actions[idx]
        else:
            return self.actions[random.randrange(self.action_size)]



    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), dtype=torch.uint8)

        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(
            state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE)
        next_state_values[non_final_mask] = self.target_net(
            non_final_next_states)

        expected_state_action_values = (
            next_state_values * GAMMA) + reward_batch

        loss = F.smooth_l1_loss(state_action_values,
                                expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def save_module(self):
        self.policy_net.save()
        self.target_net.save()
        with open(SAVE_MEMORY, 'wb') as f:
            pickle.dump(self.memory, f)
