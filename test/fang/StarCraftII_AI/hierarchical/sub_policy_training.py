import random
import numpy as np
import pandas as pd
import os
from absl import app
from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env, run_loop

DATA_FILE = 'Sub_training_data'
MORE_MINERALS_USED_REWARD_RATE = 0.00001

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
             "train_marine", 
             "train_reaper",
             "train_marauder",
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

  def get_distances(self, obs, units, xy):
    units_xy = [(unit.x, unit.y) for unit in units]
    return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1)

  def get_least_busy_building(self, building):
    return min(building, key = lambda building: building.order_length)
              

  def do_nothing(self, obs):
    return actions.RAW_FUNCTIONS.no_op()

  def get_barrackses(self, obs):
    """ get least busy barrack

    Args:
      obs
    Returns:
      int: if barrack exist, barrack.tag
      or False: if barrack not exist
    """
    completed_barrackses = self.get_my_completed_units_by_type(obs, units.Terran.Barracks)
    if len(completed_barrackses) > 0:
      return self.get_least_busy_building(completed_barrackses).tag
    else:
      return False
    
    
  def train_marine(self, obs):
    barrack_tag = self.get_barrackses(obs)
    if barrack_tag != False:
      return actions.RAW_FUNCTIONS.Train_Marine_quick("now", barrack_tag)
    else:
      return actions.RAW_FUNCTIONS.no_op()

  def train_reaper(self, obs):
    barrack_tag = self.get_barrackses(obs)
    if barrack_tag != False:
      return actions.RAW_FUNCTIONS.Train_Reaper_quick("now", barrack_tag)
    else:
      return actions.RAW_FUNCTIONS.no_op()
    
  def train_marauder(self, obs):
    barrack_tag = self.get_barrackses(obs)
    if barrack_tag != False:
      return actions.RAW_FUNCTIONS.Train_Marauder_quick("now", barrack_tag)
    else:
      return actions.RAW_FUNCTIONS.no_op()
     
  def step(self, obs):
    super(Agent, self).step(obs)
    if obs.first():
      command_center = self.get_my_units_by_type(
          obs, units.Terran.CommandCenter)[0]
      self.base_top_left = (command_center.x < 32)
      
class SubAgent_Training(Agent):

  def __init__(self):
    #print('in __init__')
    super(SubAgent_Training, self).__init__()
    self.qtable = QLearningTable(self.actions)
    if os.path.isfile(DATA_FILE + '.gz'):
      self.qtable.q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')
    self.new_game()
    
  def reset(self):
    #print('in reset')
    super(SubAgent_Training, self).reset()
    self.new_game()
    
  def new_game(self):
    #print('in new game')
    self.base_top_left = None
    self.previous_state = None
    self.previous_action = None
    self.previous_total_value_units_score = 0
    self.previous_total_value_structures_score = 0
    self.previous_total_spent_minerals = 0
    
  def get_state(self, obs):
    scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
    idle_scvs = [scv for scv in scvs if scv.order_length == 0]
    command_centers = self.get_my_units_by_type(obs, units.Terran.CommandCenter)
    supply_depots = self.get_my_units_by_type(obs, units.Terran.SupplyDepot)
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
    #print(action)

    total_value_units_score = obs.observation['score_cumulative'][3]
    total_value_structures_score = obs.observation['score_cumulative'][4]
    #print(obs.observation['score_cumulative'][11])
    total_spent_minerals = obs.observation['score_cumulative'][11]

    if self.previous_action is not None:
      step_reward = 0
      #if total_value_units_score < self.previous_total_value_units_score:
      #  step_reward -= DEAD_UNIT_REWARD_RATE * (self.previous_total_value_units_score - total_value_units_score)

      #if total_value_structures_score < self.previous_total_value_structures_score:
      #  step_reward -= DEAD_BUILDING_REWARD_RATE * (self.previous_total_value_structures_score - total_value_structures_score)
      if total_spent_minerals > self.previous_total_spent_minerals:
          step_reward += MORE_MINERALS_USED_REWARD_RATE * (total_spent_minerals - self.previous_total_spent_minerals)
        
      self.qtable.learn(self.previous_state,
                        self.previous_action,
                        obs.reward + step_reward,
                        'terminal' if obs.last() else state)

    self.previous_total_value_units_score = total_value_units_score
    self.previous_total_value_structures_score = total_value_structures_score
    self.previous_total_spent_minerals = total_spent_minerals
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


