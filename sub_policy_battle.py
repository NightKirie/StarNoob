import random
import numpy as np
import pandas as pd
import os
from types import SimpleNamespace
from functools import partial
import pickle

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




DATA_FILE = 'Sub_battle_data'
KILL_UNIT_REWARD_RATE = 0.00002
KILL_BUILDING_REWARD_RATE = 0.00004
DEAD_UNIT_REWARD_RATE = 0.00001 * 0
DEAD_BUILDING_REWARD_RATE = 0.00002 * 0
DEALT_TAKEN_RATE = 0.00001
SUB_ATTACK_DIVISION = 4
SUB_ATTACK_OFFSET = 16

BATCH_SIZE = 128
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 500

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

SAVE_POLICY_NET = 'model/battle_dqn_policy'
SAVE_TARGET_NET = 'model/battle_dqn_target'
SAVE_MEMORY = 'model/battle_memory'


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

    def do_nothing(self, obs):
        return actions.RAW_FUNCTIONS.no_op()

    def attack(self, obs, range, offset):
        armys = self.get_my_army(obs)
        if len(armys) > 0:
            attack = SimpleNamespace(**{'x': range.x * offset, 'y': range.y * offset})
            offset = SimpleNamespace(**{'x': random.randint(0, offset), 'y': random.randint(0, offset)})
            return actions.RAW_FUNCTIONS.Attack_pt(
                "now", [soldier.tag for soldier in armys], (attack.x + offset.x, attack.y + offset.y))
        return actions.RAW_FUNCTIONS.no_op()


class SubAgent_Battle(Agent):

    def __init__(self):
        super(SubAgent_Battle, self).__init__()
        self.state_size = int()             # get at step
        self.action_size = len(self.actions)
        self.policy_net = nn.Module()       # DQN init at step
        self.target_net = nn.Module()       # DQN init at step

        self.optimizer = None               # init at step
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
        self.previous_total_killed_value_units_score = 0
        self.previous_total_killed_value_structures_score = 0
        self.previous_total_damage_dealt = 0
        self.previous_total_damage_taken = 0
        self.now_reward = 0

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

        my_armys = self.get_my_army(obs)
        enemy_armys = self.get_enemy_army(obs)

        my_buildings = self.get_my_building(obs)
        enemy_buildings = self.get_enemy_building(obs)
        
        free_supply = (obs.observation.player.food_cap -
                    obs.observation.player.food_used)

        player_food_army = obs.observation.player.food_army

        return tuple([self.base_top_left,
                len(my_armys),
                len(enemy_armys),
                len(my_buildings),
                len(enemy_buildings),
                free_supply,
                player_food_army]) + \
                tuple([len(my_unit_location[i * SUB_ATTACK_DIVISION + j]) for i in range(0, SUB_ATTACK_DIVISION) for j in range(0, SUB_ATTACK_DIVISION)]) + \
                tuple([len(enemy_unit_location[i * SUB_ATTACK_DIVISION + j]) for i in range(0, SUB_ATTACK_DIVISION) for j in range(0, SUB_ATTACK_DIVISION)])

    def step(self, obs):

        super(SubAgent_Battle, self).step(obs)

        self.episode += 1
        state = self.get_state(obs)
        log.debug(f"state: {state}")
        action = self.select_action(state)
        log.info(action)

        

        if self.previous_action is not None:
            step_reward = self.get_reward(obs)
            log.log(LOG_REWARD, "battle reward = " + str(obs.reward +step_reward))
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
        total_killed_value_units_score = obs.observation.score_cumulative.killed_value_units 
        total_killed_value_structures_score = obs.observation.score_cumulative.killed_value_structures 
        total_damage_dealt = obs.observation.score_by_vital.total_damage_dealt[0]
        total_damage_taken = obs.observation.score_by_vital.total_damage_taken[0]
        prev_reward = 0
        ## Prev reward will update in this epoch
        # If kill a more valuable unit, get positive reward
        if total_killed_value_units_score > self.previous_total_killed_value_units_score:
            prev_reward += KILL_UNIT_REWARD_RATE * \
                (total_killed_value_units_score -
                    self.previous_total_killed_value_units_score)
                    
        # If kill a more valuable structure, get positive reward
        if total_killed_value_structures_score > self.previous_total_killed_value_structures_score:
            prev_reward += KILL_BUILDING_REWARD_RATE * \
                (total_killed_value_structures_score -
                    self.previous_total_killed_value_structures_score)

        # If in this epoch, dealt damage is more than taken damage, get positive reward
        if total_damage_dealt - self.previous_total_damage_dealt > total_damage_taken - self.previous_total_damage_taken:
            prev_reward += DEALT_TAKEN_RATE * \
                ((total_damage_dealt - self.previous_total_damage_dealt) - 
                    (total_damage_taken - self.previous_total_damage_taken))
        
        # If in this epoch, dealt damage is less than taken damage, get negative reward
        if total_damage_dealt - self.previous_total_damage_dealt > total_damage_taken - self.previous_total_damage_taken:
            prev_reward -= DEALT_TAKEN_RATE * \
                ((total_damage_dealt - self.previous_total_damage_dealt) - 
                    (total_damage_taken - self.previous_total_damage_taken))

        step_reward = prev_reward - self.now_reward

        ## Now reward will update in next epoch

        self.previous_total_value_units_score = total_value_units_score
        self.previous_total_value_structures_score = total_value_structures_score
        self.previous_killed_unit_score = total_killed_value_units_score
        self.previous_total_killed_value_structures_score = total_killed_value_structures_score
        self.previous_total_damage_dealt = total_damage_dealt
        self.previous_total_damage_taken = total_damage_taken
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
