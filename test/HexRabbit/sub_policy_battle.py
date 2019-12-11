import random
import numpy as np
import pandas as pd
import os
from absl import app
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env, run_loop

import base_agent

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import math
from collections import namedtuple

DATA_FILE = 'Sub_battle_data'
KILL_UNIT_REWARD_RATE = 0.00002
KILL_BUILDING_REWARD_RATE = 0.00004
DEAD_UNIT_REWARD_RATE = 0.00001 * 0
DEAD_BUILDING_REWARD_RATE = 0.00002 * 0


BATCH_SIZE = 128
GAMMA = 0.999
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

    actions = ("do_nothing",
               "attack1_1",
               "attack1_2",
               "attack1_3",
               "attack1_4",
               "attack2_1",
               "attack2_2",
               "attack2_3",
               "attack2_4",
               "attack3_1",
               "attack3_2",
               "attack3_3",
               "attack3_4",
               "attack4_1",
               "attack4_2",
               "attack4_3",
               "attack4_4",
               )

    def get_my_armys(self, obs):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type in [units.Terran.Marine, units.Terran.Reaper, units.Terran.Marauder]
                and unit.alliance == features.PlayerRelative.SELF]

    def do_nothing(self, obs):
        return actions.RAW_FUNCTIONS.no_op()

    def attack1_1(self, obs):
        armys = self.get_my_armys(obs)
        if len(armys) > 0:
            attack_xy = (8, 8)
            x_offset = random.randint(-4, 4)
            y_offset = random.randint(-4, 4)
            return actions.RAW_FUNCTIONS.Attack_pt(
                "now", [soldier.tag for soldier in armys], (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
        return actions.RAW_FUNCTIONS.no_op()

    def attack1_2(self, obs):
        armys = self.get_my_armys(obs)
        if len(armys) > 0:
            attack_xy = (24, 8)
            x_offset = random.randint(-4, 4)
            y_offset = random.randint(-4, 4)
            return actions.RAW_FUNCTIONS.Attack_pt(
                "now", [soldier.tag for soldier in armys], (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
        return actions.RAW_FUNCTIONS.no_op()

    def attack1_3(self, obs):
        armys = self.get_my_armys(obs)
        if len(armys) > 0:
            attack_xy = (40, 8)
            x_offset = random.randint(-4, 4)
            y_offset = random.randint(-4, 4)
            return actions.RAW_FUNCTIONS.Attack_pt(
                "now", [soldier.tag for soldier in armys], (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
        return actions.RAW_FUNCTIONS.no_op()

    def attack1_4(self, obs):
        armys = self.get_my_armys(obs)
        if len(armys) > 0:
            attack_xy = (56, 8)
            x_offset = random.randint(-4, 4)
            y_offset = random.randint(-4, 4)
            return actions.RAW_FUNCTIONS.Attack_pt(
                "now", [soldier.tag for soldier in armys], (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
        return actions.RAW_FUNCTIONS.no_op()

    def attack2_1(self, obs):
        armys = self.get_my_armys(obs)
        if len(armys) > 0:
            attack_xy = (8, 24)
            x_offset = random.randint(-4, 4)
            y_offset = random.randint(-4, 4)
            return actions.RAW_FUNCTIONS.Attack_pt(
                "now", [soldier.tag for soldier in armys], (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
        return actions.RAW_FUNCTIONS.no_op()

    def attack2_2(self, obs):
        armys = self.get_my_armys(obs)
        if len(armys) > 0:
            attack_xy = (24, 24)
            x_offset = random.randint(-4, 4)
            y_offset = random.randint(-4, 4)
            return actions.RAW_FUNCTIONS.Attack_pt(
                "now", [soldier.tag for soldier in armys], (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
        return actions.RAW_FUNCTIONS.no_op()

    def attack2_3(self, obs):
        armys = self.get_my_armys(obs)
        if len(armys) > 0:
            attack_xy = (40, 24)
            x_offset = random.randint(-4, 4)
            y_offset = random.randint(-4, 4)
            return actions.RAW_FUNCTIONS.Attack_pt(
                "now", [soldier.tag for soldier in armys], (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
        return actions.RAW_FUNCTIONS.no_op()

    def attack2_4(self, obs):
        armys = self.get_my_armys(obs)
        if len(armys) > 0:
            attack_xy = (56, 24)
            x_offset = random.randint(-4, 4)
            y_offset = random.randint(-4, 4)
            return actions.RAW_FUNCTIONS.Attack_pt(
                "now", [soldier.tag for soldier in armys], (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
        return actions.RAW_FUNCTIONS.no_op()

    def attack3_1(self, obs):
        armys = self.get_my_armys(obs)
        if len(armys) > 0:
            attack_xy = (8, 40)
            x_offset = random.randint(-4, 4)
            y_offset = random.randint(-4, 4)
            return actions.RAW_FUNCTIONS.Attack_pt(
                "now", [soldier.tag for soldier in armys], (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
        return actions.RAW_FUNCTIONS.no_op()

    def attack3_2(self, obs):
        armys = self.get_my_armys(obs)
        if len(armys) > 0:
            attack_xy = (24, 40)
            x_offset = random.randint(-4, 4)
            y_offset = random.randint(-4, 4)
            return actions.RAW_FUNCTIONS.Attack_pt(
                "now", [soldier.tag for soldier in armys], (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
        return actions.RAW_FUNCTIONS.no_op()

    def attack3_3(self, obs):
        armys = self.get_my_armys(obs)
        if len(armys) > 0:
            attack_xy = (40, 40)
            x_offset = random.randint(-4, 4)
            y_offset = random.randint(-4, 4)
            return actions.RAW_FUNCTIONS.Attack_pt(
                "now", [soldier.tag for soldier in armys], (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
        return actions.RAW_FUNCTIONS.no_op()

    def attack3_4(self, obs):
        armys = self.get_my_armys(obs)
        if len(armys) > 0:
            attack_xy = (56, 40)
            x_offset = random.randint(-4, 4)
            y_offset = random.randint(-4, 4)
            return actions.RAW_FUNCTIONS.Attack_pt(
                "now", [soldier.tag for soldier in armys], (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
        return actions.RAW_FUNCTIONS.no_op()

    def attack4_1(self, obs):
        armys = self.get_my_armys(obs)
        if len(armys) > 0:
            attack_xy = (8, 56)
            x_offset = random.randint(-4, 4)
            y_offset = random.randint(-4, 4)
            return actions.RAW_FUNCTIONS.Attack_pt(
                "now", [soldier.tag for soldier in armys], (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
        return actions.RAW_FUNCTIONS.no_op()

    def attack4_2(self, obs):
        armys = self.get_my_armys(obs)
        if len(armys) > 0:
            attack_xy = (24, 56)
            x_offset = random.randint(-4, 4)
            y_offset = random.randint(-4, 4)
            return actions.RAW_FUNCTIONS.Attack_pt(
                "now", [soldier.tag for soldier in armys], (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
        return actions.RAW_FUNCTIONS.no_op()

    def attack4_3(self, obs):
        armys = self.get_my_armys(obs)
        if len(armys) > 0:
            attack_xy = (40, 56)
            x_offset = random.randint(-4, 4)
            y_offset = random.randint(-4, 4)
            return actions.RAW_FUNCTIONS.Attack_pt(
                "now", [soldier.tag for soldier in armys], (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
        return actions.RAW_FUNCTIONS.no_op()

    def attack4_4(self, obs):
        armys = self.get_my_armys(obs)
        if len(armys) > 0:
            attack_xy = (56, 56)
            x_offset = random.randint(-4, 4)
            y_offset = random.randint(-4, 4)
            return actions.RAW_FUNCTIONS.Attack_pt(
                "now", [soldier.tag for soldier in armys], (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
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

    def get_state(self, obs):

        my_unit_at_1_1 = self.get_my_units_by_pos(obs, 0, 0, 16, 16)
        my_unit_at_1_2 = self.get_my_units_by_pos(obs, 16, 0, 32, 16)
        my_unit_at_1_3 = self.get_my_units_by_pos(obs, 32, 0, 48, 16)
        my_unit_at_1_4 = self.get_my_units_by_pos(obs, 48, 0, 64, 16)
        my_unit_at_2_1 = self.get_my_units_by_pos(obs, 0, 16, 16, 32)
        my_unit_at_2_2 = self.get_my_units_by_pos(obs, 16, 16, 32, 32)
        my_unit_at_2_3 = self.get_my_units_by_pos(obs, 32, 16, 48, 32)
        my_unit_at_2_4 = self.get_my_units_by_pos(obs, 48, 16, 64, 32)
        my_unit_at_3_1 = self.get_my_units_by_pos(obs, 0, 32, 16, 48)
        my_unit_at_3_2 = self.get_my_units_by_pos(obs, 16, 32, 32, 48)
        my_unit_at_3_3 = self.get_my_units_by_pos(obs, 32, 32, 48, 48)
        my_unit_at_3_4 = self.get_my_units_by_pos(obs, 48, 32, 64, 48)
        my_unit_at_4_1 = self.get_my_units_by_pos(obs, 0, 48, 16, 64)
        my_unit_at_4_2 = self.get_my_units_by_pos(obs, 16, 48, 32, 64)
        my_unit_at_4_3 = self.get_my_units_by_pos(obs, 32, 48, 48, 64)
        my_unit_at_4_4 = self.get_my_units_by_pos(obs, 48, 48, 64, 64)

        enemy_unit_at_1_1 = self.get_enemy_units_by_pos(obs, 0, 0, 16, 16)
        enemy_unit_at_1_2 = self.get_enemy_units_by_pos(obs, 16, 0, 32, 16)
        enemy_unit_at_1_3 = self.get_enemy_units_by_pos(obs, 32, 0, 48, 16)
        enemy_unit_at_1_4 = self.get_enemy_units_by_pos(obs, 48, 0, 64, 16)
        enemy_unit_at_2_1 = self.get_enemy_units_by_pos(obs, 0, 16, 16, 32)
        enemy_unit_at_2_2 = self.get_enemy_units_by_pos(obs, 16, 16, 32, 32)
        enemy_unit_at_2_3 = self.get_enemy_units_by_pos(obs, 32, 16, 48, 32)
        enemy_unit_at_2_4 = self.get_enemy_units_by_pos(obs, 48, 16, 64, 32)
        enemy_unit_at_3_1 = self.get_enemy_units_by_pos(obs, 0, 32, 16, 48)
        enemy_unit_at_3_2 = self.get_enemy_units_by_pos(obs, 16, 32, 32, 48)
        enemy_unit_at_3_3 = self.get_enemy_units_by_pos(obs, 32, 32, 48, 48)
        enemy_unit_at_3_4 = self.get_enemy_units_by_pos(obs, 48, 32, 64, 48)
        enemy_unit_at_4_1 = self.get_enemy_units_by_pos(obs, 0, 48, 16, 64)
        enemy_unit_at_4_2 = self.get_enemy_units_by_pos(obs, 16, 48, 32, 64)
        enemy_unit_at_4_3 = self.get_enemy_units_by_pos(obs, 32, 48, 48, 64)
        enemy_unit_at_4_4 = self.get_enemy_units_by_pos(obs, 48, 48, 64, 64)

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

        return (int(self.base_top_left),
                len(armys),
                free_supply,
                len(enemy_command_centers),
                len(enemy_scvs),
                len(enemy_supply_depots),
                len(enemy_barrackses),
                len(enemy_marines),
                len(my_unit_at_1_1),
                len(my_unit_at_1_2),
                len(my_unit_at_1_3),
                len(my_unit_at_1_4),
                len(my_unit_at_2_1),
                len(my_unit_at_2_2),
                len(my_unit_at_2_3),
                len(my_unit_at_2_4),
                len(my_unit_at_3_1),
                len(my_unit_at_3_2),
                len(my_unit_at_3_3),
                len(my_unit_at_3_4),
                len(my_unit_at_4_1),
                len(my_unit_at_4_2),
                len(my_unit_at_4_3),
                len(my_unit_at_4_4),
                len(enemy_unit_at_1_1),
                len(enemy_unit_at_1_2),
                len(enemy_unit_at_1_3),
                len(enemy_unit_at_1_4),
                len(enemy_unit_at_2_1),
                len(enemy_unit_at_2_2),
                len(enemy_unit_at_2_3),
                len(enemy_unit_at_2_4),
                len(enemy_unit_at_3_1),
                len(enemy_unit_at_3_2),
                len(enemy_unit_at_3_3),
                len(enemy_unit_at_3_4),
                len(enemy_unit_at_4_1),
                len(enemy_unit_at_4_2),
                len(enemy_unit_at_4_3),
                len(enemy_unit_at_4_4)
                )

    def step(self, obs):

        super(SubAgent_Battle, self).step(obs)

        self.episode += 1
        state = self.get_state(obs)
        print(state)
        action = self.select_action(state)
        print('=======================')
        print(action)
        print('=======================')

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
