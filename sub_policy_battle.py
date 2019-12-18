import random
import numpy as np
import pandas as pd
import os
import logging
from types import SimpleNamespace
from functools import partial

from absl import app
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env, run_loop

from base_agent import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import math
from collections import namedtuple
from types import SimpleNamespace
from functools import partial
from sub_policy_training import COMBAT_UNIT_NAME



DATA_FILE = 'Sub_battle_data'
KILL_UNIT_REWARD_RATE = 0.00002
KILL_BUILDING_REWARD_RATE = 0.00004
DEAD_UNIT_REWARD_RATE = 0.00001 * 0
DEAD_BUILDING_REWARD_RATE = 0.00002 * 0
SUB_ATTACK_DIVISION = 4
SUB_ATTACK_OFFSET = 16

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 500

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))



class Agent(BaseAgent):
    actions = tuple(["do_nothing"]) + \
        tuple([f"attack_{i}_{j}" for i in range(0, SUB_ATTACK_DIVISION) for j in range(0, SUB_ATTACK_DIVISION)])
    
    def __init__(self):
        super().__init__()

        for i in range(0, SUB_ATTACK_DIVISION):
            for j in range(0, SUB_ATTACK_DIVISION):
                self.__setattr__(
                    f"attack_{i}_{j}", partial(
                        self.attack, range=SimpleNamespace(**{'x': i, 'y': j}), offset=SUB_ATTACK_OFFSET))

    def get_my_armys(self, obs):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type in [getattr(units.Terran, unit) for unit in COMBAT_UNIT_NAME]
                and unit.alliance == features.PlayerRelative.SELF]

    def do_nothing(self, obs):
        return actions.RAW_FUNCTIONS.no_op()

    def attack(self, obs, range, offset):
        armys = self.get_my_armys(obs)
        if len(armys) > 0:
            attack = SimpleNamespace(**{'x': range.x * offset, 'y': range.y * offset})
            offset = SimpleNamespace(**{'x': random.randint(0, offset), 'y': random.randint(0, offset)})
            return actions.RAW_FUNCTIONS.Attack_pt(
                "now", [soldier.tag for soldier in armys], (attack.x + offset.x, attack.y + offset.y))
        return actions.RAW_FUNCTIONS.no_op()


class SubAgent_Battle(Agent):

    def __init__(self):
        super(SubAgent_Battle, self).__init__()
        self.state_size = 40
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
        super(SubAgent_Battle, self).reset()
        self.new_game()

    def new_game(self):
        self.base_top_left = None
        self.previous_state = None
        self.previous_action = None
        self.previous_total_value_units_score = 0
        self.previous_total_value_structures_score = 0
        self.previous_killed_value_units_score = 0
        self.previous_killed_value_structures_score = 0

    def get_state(self, obs=None):
        
        my_unit_location = [self.get_my_units_by_pos(obs, 
                                                    i * SUB_ATTACK_OFFSET, 
                                                    j * SUB_ATTACK_OFFSET, 
                                                    (i + 1) * SUB_ATTACK_OFFSET, 
                                                    (j + 1) * SUB_ATTACK_OFFSET) 
                                                    for i in range(0, SUB_ATTACK_DIVISION) for j in range(0, SUB_ATTACK_DIVISION)]

        enemy_unit_location = [self.get_my_units_by_pos(obs, 
                                                        i * SUB_ATTACK_OFFSET, 
                                                        j * SUB_ATTACK_OFFSET, 
                                                        (i + 1) * SUB_ATTACK_OFFSET, 
                                                        (j + 1) * SUB_ATTACK_OFFSET) 
                                                        for i in range(0, SUB_ATTACK_DIVISION) for j in range(0, SUB_ATTACK_DIVISION)]

        armys = self.get_my_armys(obs)

        free_supply = (obs.observation.player.food_cap -
                    obs.observation.player.food_used)

        enemy_scvs = self.get_enemy_units_by_type(obs, units.Terran.SCV)
        enemy_command_centers = self.get_enemy_units_by_type(
            obs, units.Terran.CommandCenter)
        enemy_supply_depots = self.get_enemy_units_by_type(
            obs, units.Terran.SupplyDepot)
        enemy_barrackses = self.get_enemy_units_by_type(
            obs, units.Terran.Barracks)
        enemy_marines = self.get_enemy_units_by_type(obs, units.Terran.Marine)

        return tuple([self.base_top_left,
                len(armys),
                free_supply,
                len(enemy_command_centers),
                len(enemy_scvs),
                len(enemy_supply_depots),
                len(enemy_barrackses),
                len(enemy_marines)]) + \
                tuple([len(my_unit_location[i * SUB_ATTACK_DIVISION + j]) for i in range(0, SUB_ATTACK_DIVISION) for j in range(0, SUB_ATTACK_DIVISION)]) + \
                tuple([len(enemy_unit_location[i * SUB_ATTACK_DIVISION + j]) for i in range(0, SUB_ATTACK_DIVISION) for j in range(0, SUB_ATTACK_DIVISION)])

    def step(self, obs):

        super(SubAgent_Battle, self).step(obs)

        self.episode += 1
        state = self.get_state(obs)
        log.debug(f"state: {state}")
        action = self.select_action(state)
        log.info(action)

        total_value_units_score = obs.observation['score_cumulative'][3]
        total_value_structures_score = obs.observation['score_cumulative'][4]
        killed_value_units_score = obs.observation['score_cumulative'][5]
        killed_value_structures_score = obs.observation['score_cumulative'][6]
        self.previous_total_value_units = 0
        self.previous_total_value_structures = 0

        if self.previous_action is not None:
            step_reward = 0

            if killed_value_units_score > self.previous_killed_value_units_score:
                step_reward += KILL_UNIT_REWARD_RATE * \
                    (killed_value_units_score -
                     self.previous_killed_value_units_score)

            if killed_value_structures_score > self.previous_killed_value_structures_score:
                step_reward += KILL_BUILDING_REWARD_RATE * \
                    (killed_value_structures_score -
                     self.previous_killed_value_structures_score)
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
        self.previous_killed_unit_score = killed_value_units_score
        self.previous_killed_value_structures_score = killed_value_structures_score
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

    def save_module(self):
        pass

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
