import random
import numpy as np
import pandas as pd
import os
from absl import app
from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env, run_loop

DATA_FILE = 'Sub_battle_data'
KILL_UNIT_REWARD_RATE = 0.00002
KILL_BUILDING_REWARD_RATE = 0.00004
DEAD_UNIT_REWARD_RATE = 0.00001 * 0
DEAD_BUILDING_REWARD_RATE = 0.00002 * 0

class QLearningTable:
  def __init__(self, actions, learning_rate=0.01, reward_decay=0.9):
    self.actions = actions
    self.learning_rate = learning_rate
    self.reward_decay = reward_decay
    self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

  def choose_action(self, observation, e_greedy=0.9):
    self.check_state_exist(observation)
    if np.random.uniform() < e_greedy:
      state_action = self.q_table.loc[observation, :]
      action = np.random.choice(
          state_action[state_action == np.max(state_action)].index)
    else:
      action = np.random.choice(self.actions)
    return action

  def learn(self, s, a, r, s_):
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

  def get_my_units_by_type(self, obs, unit_type):
    return [unit for unit in obs.observation.raw_units
            if unit.unit_type == unit_type 
            and unit.alliance == features.PlayerRelative.SELF]
  
  def get_enemy_units_by_type(self, obs, unit_type):
    return [unit for unit in obs.observation.raw_units
            if unit.unit_type == unit_type 
            and unit.alliance == features.PlayerRelative.ENEMY]

  def get_my_units_by_pos(self, obs, pos1x, pos1y, pos2x, pos2y):
    return  [unit for unit in obs.observation.raw_units
             if unit.alliance == features.PlayerRelative.SELF
             and unit.x >= pos1x and unit.x < pos2x
             and unit.y >= pos1y and unit.y < pos2y]

  def get_enemy_units_by_pos(self, obs, pos1x, pos1y, pos2x, pos2y):
    return  [unit for unit in obs.observation.raw_units
             if unit.alliance == features.PlayerRelative.ENEMY
             and unit.x >= pos1x and unit.x < pos2x
             and unit.y >= pos1y and unit.y < pos2y]
  
  def get_my_completed_units_by_type(self, obs, unit_type):
    return [unit for unit in obs.observation.raw_units
            if unit.unit_type == unit_type 
            and unit.build_progress == 100
            and unit.alliance == features.PlayerRelative.SELF]
    
  def get_enemy_completed_units_by_type(self, obs, unit_type):
    return [unit for unit in obs.observation.raw_units
            if unit.unit_type == unit_type 
            and unit.build_progress == 100
            and unit.alliance == features.PlayerRelative.ENEMY]

  def do_nothing(self, obs):
    return actions.RAW_FUNCTIONS.no_op()

  def attack1_1(self, obs):
    marines = self.get_my_units_by_type(obs, units.Terran.Marine)
    if len(marines) > 0:
      attack_xy = (8, 8)
      x_offset = random.randint(-4, 4)
      y_offset = random.randint(-4, 4)
      return actions.RAW_FUNCTIONS.Attack_pt(
          "now", [soldier.tag for soldier in marines], (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
    return actions.RAW_FUNCTIONS.no_op()

  def attack1_2(self, obs):
    marines = self.get_my_units_by_type(obs, units.Terran.Marine)
    if len(marines) > 0:
      attack_xy = (24, 8)
      x_offset = random.randint(-4, 4)
      y_offset = random.randint(-4, 4)
      return actions.RAW_FUNCTIONS.Attack_pt(
          "now", [soldier.tag for soldier in marines], (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
    return actions.RAW_FUNCTIONS.no_op()

  def attack1_3(self, obs):
    marines = self.get_my_units_by_type(obs, units.Terran.Marine)
    if len(marines) > 0:
      attack_xy = (40, 8)
      x_offset = random.randint(-4, 4)
      y_offset = random.randint(-4, 4)
      return actions.RAW_FUNCTIONS.Attack_pt(
          "now", [soldier.tag for soldier in marines], (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
    return actions.RAW_FUNCTIONS.no_op()

  def attack1_4(self, obs):
    marines = self.get_my_units_by_type(obs, units.Terran.Marine)
    if len(marines) > 0:
      attack_xy = (56, 8)
      x_offset = random.randint(-4, 4)
      y_offset = random.randint(-4, 4)
      return actions.RAW_FUNCTIONS.Attack_pt(
          "now", [soldier.tag for soldier in marines], (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
    return actions.RAW_FUNCTIONS.no_op()

  def attack2_1(self, obs):
    marines = self.get_my_units_by_type(obs, units.Terran.Marine)
    if len(marines) > 0:
      attack_xy = (8, 24)
      x_offset = random.randint(-4, 4)
      y_offset = random.randint(-4, 4)
      return actions.RAW_FUNCTIONS.Attack_pt(
          "now", [soldier.tag for soldier in marines], (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
    return actions.RAW_FUNCTIONS.no_op()

  def attack2_2(self, obs):
    marines = self.get_my_units_by_type(obs, units.Terran.Marine)
    if len(marines) > 0:
      attack_xy = (24, 24)
      x_offset = random.randint(-4, 4)
      y_offset = random.randint(-4, 4)
      return actions.RAW_FUNCTIONS.Attack_pt(
          "now", [soldier.tag for soldier in marines], (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
    return actions.RAW_FUNCTIONS.no_op()

  def attack2_3(self, obs):
    marines = self.get_my_units_by_type(obs, units.Terran.Marine)
    if len(marines) > 0:
      attack_xy = (40, 24)
      x_offset = random.randint(-4, 4)
      y_offset = random.randint(-4, 4)
      return actions.RAW_FUNCTIONS.Attack_pt(
          "now", [soldier.tag for soldier in marines], (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
    return actions.RAW_FUNCTIONS.no_op()

  def attack2_4(self, obs):
    marines = self.get_my_units_by_type(obs, units.Terran.Marine)
    if len(marines) > 0:
      attack_xy = (56, 24)
      x_offset = random.randint(-4, 4)
      y_offset = random.randint(-4, 4)
      return actions.RAW_FUNCTIONS.Attack_pt(
          "now", [soldier.tag for soldier in marines], (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
    return actions.RAW_FUNCTIONS.no_op()
  
  def attack3_1(self, obs):
    marines = self.get_my_units_by_type(obs, units.Terran.Marine)
    if len(marines) > 0:
      attack_xy = (8, 40)
      x_offset = random.randint(-4, 4)
      y_offset = random.randint(-4, 4)
      return actions.RAW_FUNCTIONS.Attack_pt(
          "now", [soldier.tag for soldier in marines], (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
    return actions.RAW_FUNCTIONS.no_op()

  def attack3_2(self, obs):
    marines = self.get_my_units_by_type(obs, units.Terran.Marine)
    if len(marines) > 0:
      attack_xy = (24, 40)
      x_offset = random.randint(-4, 4)
      y_offset = random.randint(-4, 4)
      return actions.RAW_FUNCTIONS.Attack_pt(
          "now", [soldier.tag for soldier in marines], (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
    return actions.RAW_FUNCTIONS.no_op()

  def attack3_3(self, obs):
    marines = self.get_my_units_by_type(obs, units.Terran.Marine)
    if len(marines) > 0:
      attack_xy = (40, 40)
      x_offset = random.randint(-4, 4)
      y_offset = random.randint(-4, 4)
      return actions.RAW_FUNCTIONS.Attack_pt(
          "now", [soldier.tag for soldier in marines], (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
    return actions.RAW_FUNCTIONS.no_op()

  def attack3_4(self, obs):
    marines = self.get_my_units_by_type(obs, units.Terran.Marine)
    if len(marines) > 0:
      attack_xy = (56, 40)
      x_offset = random.randint(-4, 4)
      y_offset = random.randint(-4, 4)
      return actions.RAW_FUNCTIONS.Attack_pt(
          "now", [soldier.tag for soldier in marines], (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
    return actions.RAW_FUNCTIONS.no_op()

  def attack4_1(self, obs):
    marines = self.get_my_units_by_type(obs, units.Terran.Marine)
    if len(marines) > 0:
      attack_xy = (8, 56)
      x_offset = random.randint(-4, 4)
      y_offset = random.randint(-4, 4)
      return actions.RAW_FUNCTIONS.Attack_pt(
          "now", [soldier.tag for soldier in marines], (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
    return actions.RAW_FUNCTIONS.no_op()

  def attack4_2(self, obs):
    marines = self.get_my_units_by_type(obs, units.Terran.Marine)
    if len(marines) > 0:
      attack_xy = (24, 56)
      x_offset = random.randint(-4, 4)
      y_offset = random.randint(-4, 4)
      return actions.RAW_FUNCTIONS.Attack_pt(
          "now", [soldier.tag for soldier in marines], (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
    return actions.RAW_FUNCTIONS.no_op()

  def attack4_3(self, obs):
    marines = self.get_my_units_by_type(obs, units.Terran.Marine)
    if len(marines) > 0:
      attack_xy = (40, 56)
      x_offset = random.randint(-4, 4)
      y_offset = random.randint(-4, 4)
      return actions.RAW_FUNCTIONS.Attack_pt(
          "now", [soldier.tag for soldier in marines], (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
    return actions.RAW_FUNCTIONS.no_op()

  def attack4_4(self, obs):
    marines = self.get_my_units_by_type(obs, units.Terran.Marine)
    if len(marines) > 0:
      attack_xy = (56, 56)
      x_offset = random.randint(-4, 4)
      y_offset = random.randint(-4, 4)
      return actions.RAW_FUNCTIONS.Attack_pt(
          "now", [soldier.tag for soldier in marines], (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
    return actions.RAW_FUNCTIONS.no_op()

  def step(self, obs):
    super(Agent, self).step(obs)
    if obs.first():
      command_center = self.get_my_units_by_type(
          obs, units.Terran.CommandCenter)[0]
      self.base_top_left = (command_center.x < 32)

class SubAgent_Battle(Agent):

  def __init__(self):
    super(SubAgent_Battle, self).__init__()
    self.qtable = QLearningTable(self.actions)
    if os.path.isfile(DATA_FILE + '.gz'):
      self.qtable.q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')
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

    marines = self.get_my_units_by_type(obs, units.Terran.Marine) 

    free_supply = (obs.observation.player.food_cap - 
                   obs.observation.player.food_used)
    
    enemy_scvs = self.get_enemy_units_by_type(obs, units.Terran.SCV)
    enemy_command_centers = self.get_enemy_units_by_type(
        obs, units.Terran.CommandCenter)
    enemy_supply_depots = self.get_enemy_units_by_type(
        obs, units.Terran.SupplyDepot)
    enemy_barrackses = self.get_enemy_units_by_type(obs, units.Terran.Barracks)
    enemy_marines = self.get_enemy_units_by_type(obs, units.Terran.Marine)
    

    return (self.base_top_left,
            len(marines),
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
      
    if obs.last():
      self.qtable.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')
    super(SubAgent_Battle, self).step(obs)
    state = str(self.get_state(obs))
    action = self.qtable.choose_action(state)
    #print(action)

    total_value_units_score = obs.observation['score_cumulative'][3]
    total_value_structures_score = obs.observation['score_cumulative'][4]
    killed_value_units_score = obs.observation['score_cumulative'][5]
    killed_value_structures_score = obs.observation['score_cumulative'][6]
    self.previous_total_value_units = 0
    self.previous_total_value_structures = 0

    if self.previous_action is not None:
      step_reward = 0
      #if total_value_units_score < self.previous_total_value_units_score:
      #  step_reward -= DEAD_UNIT_REWARD_RATE * (self.previous_total_value_units_score - total_value_units_score)

      #if total_value_structures_score < self.previous_total_value_structures_score:
      #  step_reward -= DEAD_BUILDING_REWARD_RATE * (self.previous_total_value_structures_score - total_value_structures_score)

      if killed_value_units_score > self.previous_killed_value_units_score:
          step_reward += KILL_UNIT_REWARD_RATE * (killed_value_units_score - self.previous_killed_value_units_score)
              
      if killed_value_structures_score > self.previous_killed_value_structures_score:
          step_reward += KILL_BUILDING_REWARD_RATE * (killed_value_structures_score - self.previous_killed_value_structures_score)
      
      self.qtable.learn(self.previous_state,
                        self.previous_action,
                        obs.reward + step_reward,
                        'terminal' if obs.last() else state)

    self.previous_total_value_units_score = total_value_units_score
    self.previous_total_value_structures_score = total_value_structures_score
    self.previous_killed_unit_score = killed_value_units_score
    self.previous_killed_value_structures_score = killed_value_structures_score
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