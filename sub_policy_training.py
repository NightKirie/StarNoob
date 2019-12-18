import random
import numpy as np
import pandas as pd
import os
from absl import app
from functools import partial

from pysc2.lib import actions, features, units
from pysc2.env import sc2_env, run_loop

#from base_agent import QLearningTable
import base_agent

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from collections import namedtuple


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

BATCH_SIZE = 128
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 500

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

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
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        print(state_size)
        print(action_size)
        self.net = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, x):
        return self.net(x)

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
        self.state_size = 41
        self.action_size = len(self.actions)
        self.policy_net = DQN(self.state_size, self.action_size)
        self.target_net = DQN(self.state_size, self.action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)

        self.episode = 0
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
        reapers = self.get_my_units_by_type(obs, units.Terran.Reaper)
        marauders = self.get_my_units_by_type(obs, units.Terran.Marauder)
        ghosts = self.get_my_units_by_type(obs, units.Terran.Ghost)
        hellions = self.get_my_units_by_type(obs, units.Terran.Hellion)
        siegetanks = self.get_my_units_by_type(obs, units.Terran.SiegeTank)
        widowmines = self.get_my_units_by_type(obs, units.Terran.WidowMine)
        hellbats = self.get_my_units_by_type(obs, units.Terran.Hellbat)
        thors = self.get_my_units_by_type(obs, units.Terran.Thor)
        liberators = self.get_my_units_by_type(obs, units.Terran.Liberator)
        cyclones = self.get_my_units_by_type(obs, units.Terran.Cyclone)
        vikingfighters = self.get_my_units_by_type(obs, units.Terran.VikingFighter)
        medivacs = self.get_my_units_by_type(obs, units.Terran.Medivac)
        ravens = self.get_my_units_by_type(obs, units.Terran.Raven)
        banshees = self.get_my_units_by_type(obs, units.Terran.Banshee)
        battlecruisers = self.get_my_units_by_type(obs, units.Terran.Battlecruiser)
        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)

        can_afford_marine = obs.observation.player.minerals >= 100
        can_afford_reapers = obs.observation.player.minerals >= 50 and obs.observation.player.vespene >= 50
        can_afford_marauders = obs.observation.player.minerals >= 100 and obs.observation.player.vespene >= 25
        can_afford_ghosts = obs.observation.player.minerals >= 150 and obs.observation.player.vespene >= 125
        can_afford_hellionsv = obs.observation.player.minerals >= 100
        can_afford_siegetanks = obs.observation.player.minerals >= 150 and obs.observation.player.vespene >= 125
        can_afford_widowmines = obs.observation.player.minerals >= 75 and obs.observation.player.vespene >= 25
        can_afford_hellbats = obs.observation.player.minerals >= 100
        can_afford_thors = obs.observation.player.minerals >= 300 and obs.observation.player.vespene >= 200
        can_afford_liberators = obs.observation.player.minerals >= 150 and obs.observation.player.vespene >= 150
        can_afford_cyclones = obs.observation.player.minerals >= 150 and obs.observation.player.vespene >= 100
        can_afford_vikingfighters = obs.observation.player.minerals >= 150 and obs.observation.player.vespene >= 75
        can_afford_medivacs = obs.observation.player.minerals >= 100 and obs.observation.player.vespene >= 100
        can_afford_ravens = obs.observation.player.minerals >= 100 and obs.observation.player.vespene >= 200
        can_afford_banshees = obs.observation.player.minerals >= 150 and obs.observation.player.vespene >= 100
        can_afford_battlecruisers = obs.observation.player.minerals >= 400 and obs.observation.player.vespene >= 300

        

        return (self.base_top_left,
                len(command_centers),
                len(scvs),
                len(idle_scvs),
                len(supply_depots),
                len(completed_supply_depots),
                len(barrackses),
                len(completed_barrackses),
                len(marines),
                len(reapers),
                len(marauders),
                len(ghosts),
                len(hellions),
                len(siegetanks),
                len(widowmines),
                len(hellbats),
                len(thors),
                len(liberators),
                len(cyclones),
                len(vikingfighters),
                len(medivacs),
                len(ravens),
                len(banshees),
                len(battlecruisers),
                free_supply,
                can_afford_marine,
                can_afford_reapers,
                can_afford_marauders,
                can_afford_ghosts,
                can_afford_hellionsv,
                can_afford_siegetanks,
                can_afford_widowmines,
                can_afford_hellbats,
                can_afford_thors,
                can_afford_liberators,
                can_afford_cyclones,
                can_afford_vikingfighters,
                can_afford_medivacs,
                can_afford_ravens,
                can_afford_banshees,
                can_afford_battlecruisers,
                )

    def step(self, obs):
        super(SubAgent_Training, self).step(obs)

        self.episode += 1
        state = self.get_state(obs)
        print(state)
        action = self.select_action(state)
        print('=======================')
        print(action)
        print('=======================')


        # if obs.last():
        #     self.qtable.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')
        # super(SubAgent_Training, self).step(obs)
        # state = str(self.get_state(obs))
        # action = self.qtable.choose_action(state)
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

            # self.qtable.learn(self.previous_state,
            #                   self.previous_action,
            #                   obs.reward + step_reward,
            #                   'terminal' if obs.last() else state)
            if not obs.last:
                self.memory.push(self.previous_state,
                                 self.previous_action,
                                 state,
                                 obs.reward + step_reward)

                self.optimize_model()
            else:
                pass
        
        if self.episode % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.previous_total_value_units_score = total_value_units_score
        self.previous_total_value_structures_score = total_value_structures_score
        self.previous_total_spent_minerals = total_spent_minerals
        self.previous_total_spent_vespene = total_spent_vespene
        self.previous_state = state
        self.previous_action = action
        return getattr(self, action)(obs)

    def set_top_left(self, obs):
        if obs.first():
            command_center = self.get_my_units_by_type(
                obs, units.Terran.CommandCenter)[0]
            self.base_top_left = (command_center.x < 32)

    def select_action(self, state):
        sample = random.random()
        eps_threshold = 0.1
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
        pass#self.qtable.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')