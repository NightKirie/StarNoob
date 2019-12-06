import random
import numpy as np
import pandas as pd
import os
from absl import app
from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env, run_loop

DATA_FILE = 'Sub_building_data'
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
             "harvest_minerals", 
             "build_supply_depot", 
             "build_barracks", 
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
    order_array = []
    for item in building:
      order_array.append(item.order_length)
    return order_array.index(min(order_array))
              

  def do_nothing(self, obs):
    return actions.RAW_FUNCTIONS.no_op()

  def harvest_minerals(self, obs):
    scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
    idle_scvs = [scv for scv in scvs if scv.order_length == 0]
    if len(idle_scvs) > 0:
      mineral_patches = [unit for unit in obs.observation.raw_units
                         if unit.unit_type in [
                           units.Neutral.BattleStationMineralField,
                           units.Neutral.BattleStationMineralField750,
                           units.Neutral.LabMineralField,
                           units.Neutral.LabMineralField750,
                           units.Neutral.MineralField,
                           units.Neutral.MineralField750,
                           units.Neutral.PurifierMineralField,
                           units.Neutral.PurifierMineralField750,
                           units.Neutral.PurifierRichMineralField,
                           units.Neutral.PurifierRichMineralField750,
                           units.Neutral.RichMineralField,
                           units.Neutral.RichMineralField750
                         ]]
      scv = random.choice(idle_scvs)
      distances = self.get_distances(obs, mineral_patches, (scv.x, scv.y))
      mineral_patch = mineral_patches[np.argmin(distances)] 
      return actions.RAW_FUNCTIONS.Harvest_Gather_unit(
          "now", scv.tag, mineral_patch.tag)
    return actions.RAW_FUNCTIONS.no_op()

  def build_supply_depot(self, obs):
    supply_depots = self.get_my_units_by_type(obs, units.Terran.SupplyDepot)
    scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
    if (len(supply_depots) == 0 and obs.observation.player.minerals >= 100 and
        len(scvs) > 0):
      supply_depot_xy = (22, 26) if self.base_top_left else (35, 42)
      distances = self.get_distances(obs, scvs, supply_depot_xy)
      scv = scvs[np.argmin(distances)]
      return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt(
          "now", scv.tag, supply_depot_xy)
    if (len(supply_depots) == 1 and obs.observation.player.minerals >= 100 and
        len(scvs) > 0):
      supply_depot_xy = (21, 26) if self.base_top_left else (36, 42)
      distances = self.get_distances(obs, scvs, supply_depot_xy)
      scv = scvs[np.argmin(distances)]
      return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt(
          "now", scv.tag, supply_depot_xy)
    if (len(supply_depots) == 2 and obs.observation.player.minerals >= 100 and
        len(scvs) > 0):
      supply_depot_xy = (19, 26) if self.base_top_left else (38, 42)
      distances = self.get_distances(obs, scvs, supply_depot_xy)
      scv = scvs[np.argmin(distances)]
      return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt(
          "now", scv.tag, supply_depot_xy)
      
    return actions.RAW_FUNCTIONS.no_op()


  def build_barracks(self, obs):
    completed_supply_depots = self.get_my_completed_units_by_type(
        obs, units.Terran.SupplyDepot)
    barrackses = self.get_my_units_by_type(obs, units.Terran.Barracks)
    scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
    if (len(completed_supply_depots) > 0 and len(barrackses) == 0 and 
        obs.observation.player.minerals >= 150 and len(scvs) > 0):
      barracks_xy = (22, 21) if self.base_top_left else (35, 45)
      distances = self.get_distances(obs, scvs, barracks_xy)
      scv = scvs[np.argmin(distances)]
      return actions.RAW_FUNCTIONS.Build_Barracks_pt(
          "now", scv.tag, barracks_xy)
    if (len(completed_supply_depots) > 0 and len(barrackses) == 1 and 
        obs.observation.player.minerals >= 150 and len(scvs) > 0):
      barracks_xy = (22, 24) if self.base_top_left else (35, 48)
      distances = self.get_distances(obs, scvs, barracks_xy)
      scv = scvs[np.argmin(distances)]
      return actions.RAW_FUNCTIONS.Build_Barracks_pt(
          "now", scv.tag, barracks_xy)
    return actions.RAW_FUNCTIONS.no_op()


  def train_marine(self, obs):
    completed_barrackses = self.get_my_completed_units_by_type(
        obs, units.Terran.Barracks)
    free_supply = (obs.observation.player.food_cap - 
                   obs.observation.player.food_used)
    if (len(completed_barrackses) > 0 and obs.observation.player.minerals >= 100
        and free_supply > 0):
      barracks = self.get_my_units_by_type(obs, units.Terran.Barracks)
      barracks = barracks[self.get_least_busy_building(barracks)]
      if barracks.order_length < 5:
        return actions.RAW_FUNCTIONS.Train_Marine_quick("now", barracks.tag)
    return actions.RAW_FUNCTIONS.no_op()

  def step(self, obs):
    super(Agent, self).step(obs)
    if obs.first():
      command_center = self.get_my_units_by_type(
          obs, units.Terran.CommandCenter)[0]
      self.base_top_left = (command_center.x < 32)

class SubAgent_Economic(Agent):

  def __init__(self):
    #print('in __init__')
    super(SubAgent_Economic, self).__init__()
    self.qtable = QLearningTable(self.actions)
    if os.path.isfile(DATA_FILE + '.gz'):
      self.qtable.q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')
    self.new_game()
    
  def reset(self):
    #print('in reset')
    super(SubAgent_Economic, self).reset()
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
    can_afford_supply_depot = obs.observation.player.minerals >= 100
    can_afford_barracks = obs.observation.player.minerals >= 150

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
            can_afford_supply_depot,
            can_afford_barracks
            )
    
  def step(self, obs):
      
    if obs.last():
      self.qtable.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')
      self.qtable.q_table.to_csv(DATA_FILE + '.csv')
    super(SubAgent_Economic, self).step(obs)
    state = str(self.get_state(obs))
    action = self.qtable.choose_action(state)
    #print(action)

    total_value_units_score = obs.observation['score_cumulative'][3]
    total_value_structures_score = obs.observation['score_cumulative'][4]
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
    self.qtable.q_table.to_csv(DATA_FILE + '.csv')

  def set_top_left(self, obs):
    if obs.first():
      command_center = self.get_my_units_by_type(
          obs, units.Terran.CommandCenter)[0]
      self.base_top_left = (command_center.x < 32)
