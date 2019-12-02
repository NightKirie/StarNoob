import random
import numpy as np
import pandas as pd
import os
from absl import app
from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env, run_loop

DATA_FILE = 'AI_agent_data'
KILL_UNIT_REWARD_RATE = 0.00002
KILL_BUILDING_REWARD_RATE = 0.00004
DEAD_UNIT_REWARD_RATE = 0.00001 * 0
DEAD_BUILDING_REWARD_RATE = 0.00002 * 0
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
             "train_marine", 
             "attack1_1",
             "attack1_2",
             "attack1_3",
             "attack2_1",
             "attack2_2",
             "attack2_3",
             "attack3_1",
             "attack3_2",
             "attack3_3",
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

  def attack1_1(self, obs):
    marines = self.get_my_units_by_type(obs, units.Terran.Marine)
    if len(marines) > 0:
      attack_xy = (11, 11)
      distances = self.get_distances(obs, marines, attack_xy)
      #marine = marines[np.argmax(distances)]
      x_offset = random.randint(-5, 5)
      y_offset = random.randint(-5, 5)
      return actions.RAW_FUNCTIONS.Attack_pt(
          "now", [soldier.tag for soldier in marines], (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
    return actions.RAW_FUNCTIONS.no_op()

  def attack1_2(self, obs):
    marines = self.get_my_units_by_type(obs, units.Terran.Marine)
    if len(marines) > 0:
      attack_xy = (32, 11)
      distances = self.get_distances(obs, marines, attack_xy)
      #marine = marines[np.argmax(distances)]
      x_offset = random.randint(-5, 5)
      y_offset = random.randint(-5, 5)
      return actions.RAW_FUNCTIONS.Attack_pt(
          "now", [soldier.tag for soldier in marines], (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
    return actions.RAW_FUNCTIONS.no_op()

  def attack1_3(self, obs):
    marines = self.get_my_units_by_type(obs, units.Terran.Marine)
    if len(marines) > 0:
      attack_xy = (53, 11)
      distances = self.get_distances(obs, marines, attack_xy)
      #marine = marines[np.argmax(distances)]
      x_offset = random.randint(-5, 5)
      y_offset = random.randint(-5, 5)
      return actions.RAW_FUNCTIONS.Attack_pt(
          "now", [soldier.tag for soldier in marines], (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
    return actions.RAW_FUNCTIONS.no_op()

  def attack2_1(self, obs):
    marines = self.get_my_units_by_type(obs, units.Terran.Marine)
    if len(marines) > 0:
      attack_xy = (11, 32)
      distances = self.get_distances(obs, marines, attack_xy)
      #marine = marines[np.argmax(distances)]
      x_offset = random.randint(-5, 5)
      y_offset = random.randint(-5, 5)
      return actions.RAW_FUNCTIONS.Attack_pt(
          "now", [soldier.tag for soldier in marines], (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
    return actions.RAW_FUNCTIONS.no_op()

  def attack2_2(self, obs):
    marines = self.get_my_units_by_type(obs, units.Terran.Marine)
    if len(marines) > 0:
      attack_xy = (32, 32)
      distances = self.get_distances(obs, marines, attack_xy)
      #marine = marines[np.argmax(distances)]
      x_offset = random.randint(-5, 5)
      y_offset = random.randint(-5, 5)
      return actions.RAW_FUNCTIONS.Attack_pt(
          "now", [soldier.tag for soldier in marines], (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
    return actions.RAW_FUNCTIONS.no_op()

  def attack2_3(self, obs):
    marines = self.get_my_units_by_type(obs, units.Terran.Marine)
    if len(marines) > 0:
      attack_xy = (53, 32)
      distances = self.get_distances(obs, marines, attack_xy)
      #marine = marines[np.argmax(distances)]
      x_offset = random.randint(-5, 5)
      y_offset = random.randint(-5, 5)
      return actions.RAW_FUNCTIONS.Attack_pt(
          "now", [soldier.tag for soldier in marines], (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
    return actions.RAW_FUNCTIONS.no_op()
  
  def attack3_1(self, obs):
    marines = self.get_my_units_by_type(obs, units.Terran.Marine)
    if len(marines) > 0:
      attack_xy = (11, 53)
      distances = self.get_distances(obs, marines, attack_xy)
      #marine = marines[np.argmax(distances)]
      x_offset = random.randint(-5, 5)
      y_offset = random.randint(-5, 5)
      return actions.RAW_FUNCTIONS.Attack_pt(
          "now", [soldier.tag for soldier in marines], (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
    return actions.RAW_FUNCTIONS.no_op()

  def attack3_2(self, obs):
    marines = self.get_my_units_by_type(obs, units.Terran.Marine)
    if len(marines) > 0:
      attack_xy = (32, 53)
      distances = self.get_distances(obs, marines, attack_xy)
      #marine = marines[np.argmax(distances)]
      x_offset = random.randint(-5, 5)
      y_offset = random.randint(-5, 5)
      return actions.RAW_FUNCTIONS.Attack_pt(
          "now", [soldier.tag for soldier in marines], (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
    return actions.RAW_FUNCTIONS.no_op()

  def attack3_3(self, obs):
    marines = self.get_my_units_by_type(obs, units.Terran.Marine)
    if len(marines) > 0:
      attack_xy = (53, 53)
      distances = self.get_distances(obs, marines, attack_xy)
      #marine = marines[np.argmax(distances)]
      x_offset = random.randint(-5, 5)
      y_offset = random.randint(-5, 5)
      return actions.RAW_FUNCTIONS.Attack_pt(
          "now", [soldier.tag for soldier in marines], (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
    return actions.RAW_FUNCTIONS.no_op()

  def step(self, obs):
    super(Agent, self).step(obs)
    if obs.first():
      command_center = self.get_my_units_by_type(
          obs, units.Terran.CommandCenter)[0]
      self.base_top_left = (command_center.x < 32)
      
class RandomAgent(Agent):
  def step(self, obs):
    super(RandomAgent, self).step(obs)
    action = random.choice(self.actions)
    return getattr(self, action)(obs)

class SmartAgent(Agent):

  def __init__(self):
    super(SmartAgent, self).__init__()
    self.qtable = QLearningTable(self.actions)
    if os.path.isfile(DATA_FILE + '.gz'):
      self.qtable.q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')
    self.new_game()
    
  def reset(self):
    super(SmartAgent, self).reset()
    self.new_game()
    
  def new_game(self):
    self.base_top_left = None
    self.previous_state = None
    self.previous_action = None
    self.previous_total_value_units_score = 0
    self.previous_total_value_structures_score = 0
    self.previous_killed_value_units_score = 0
    self.previous_killed_value_structures_score = 0
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

    my_unit_at_1_1 = self.get_my_units_by_pos(obs, 0, 0, 21, 21)
    my_unit_at_1_2 = self.get_my_units_by_pos(obs, 21, 0, 42, 21)
    my_unit_at_1_3 = self.get_my_units_by_pos(obs, 42, 0, 64, 21)
    my_unit_at_2_1 = self.get_my_units_by_pos(obs, 0, 21, 21, 42)
    my_unit_at_2_2 = self.get_my_units_by_pos(obs, 21, 21, 42, 42)
    my_unit_at_2_3 = self.get_my_units_by_pos(obs, 42, 21, 64, 42)
    my_unit_at_3_1 = self.get_my_units_by_pos(obs, 0, 42, 21, 64)
    my_unit_at_3_2 = self.get_my_units_by_pos(obs, 21, 42, 42, 64)
    my_unit_at_3_3 = self.get_my_units_by_pos(obs, 42, 42, 64, 64)
    
    enemy_unit_at_1_1 = self.get_enemy_units_by_pos(obs, 0, 0, 21, 21)
    enemy_unit_at_1_2 = self.get_enemy_units_by_pos(obs, 21, 0, 42, 21)
    enemy_unit_at_1_3 = self.get_enemy_units_by_pos(obs, 42, 0, 64, 21)
    enemy_unit_at_2_1 = self.get_enemy_units_by_pos(obs, 0, 21, 21, 42)
    enemy_unit_at_2_2 = self.get_enemy_units_by_pos(obs, 21, 21, 42, 42)
    enemy_unit_at_2_3 = self.get_enemy_units_by_pos(obs, 42, 21, 64, 42)
    enemy_unit_at_3_1 = self.get_enemy_units_by_pos(obs, 0, 42, 21, 64)
    enemy_unit_at_3_2 = self.get_enemy_units_by_pos(obs, 21, 42, 42, 64)
    enemy_unit_at_3_3 = self.get_enemy_units_by_pos(obs, 42, 42, 64, 64)

    #print(len(enemy_unit_at_3_3))
    #print([enemy_unit_at_1_1, enemy_unit_at_2_2, enemy_unit_at_3_3])
    #print([len(enemy_unit_at_1_1), len(enemy_unit_at_1_2), len(enemy_unit_at_1_3), len(enemy_unit_at_2_1), len(enemy_unit_at_2_2), len(enemy_unit_at_2_3), len(enemy_unit_at_3_1), len(enemy_unit_at_3_2), len(enemy_unit_at_3_3)])
    
    marines = self.get_my_units_by_type(obs, units.Terran.Marine) 
    
    #queued_marines = (completed_barrackses[0].order_length 
    #                  if len(completed_barrackses) > 0 else 0)
    
    free_supply = (obs.observation.player.food_cap - 
                   obs.observation.player.food_used)
    can_afford_supply_depot = obs.observation.player.minerals >= 100
    can_afford_barracks = obs.observation.player.minerals >= 150
    can_afford_marine = obs.observation.player.minerals >= 100
    
    enemy_scvs = self.get_enemy_units_by_type(obs, units.Terran.SCV)
    #enemy_idle_scvs = [scv for scv in enemy_scvs if scv.order_length == 0]
    enemy_command_centers = self.get_enemy_units_by_type(
        obs, units.Terran.CommandCenter)
    enemy_supply_depots = self.get_enemy_units_by_type(
        obs, units.Terran.SupplyDepot)
    #enemy_completed_supply_depots = self.get_enemy_completed_units_by_type(
    #    obs, units.Terran.SupplyDepot)
    enemy_barrackses = self.get_enemy_units_by_type(obs, units.Terran.Barracks)
    #enemy_completed_barrackses = self.get_enemy_completed_units_by_type(
    #    obs, units.Terran.Barracks)
    enemy_marines = self.get_enemy_units_by_type(obs, units.Terran.Marine)
    

    return (self.base_top_left,
            len(command_centers),
            len(scvs),
            len(idle_scvs),
            len(supply_depots),
            len(completed_supply_depots),
            len(barrackses),
            len(completed_barrackses),
            len(marines),
            #queued_marines,
            free_supply,
            can_afford_supply_depot,
            can_afford_barracks,
            can_afford_marine,
            len(enemy_command_centers),
            len(enemy_scvs),
            #len(enemy_idle_scvs),
            len(enemy_supply_depots),
            #len(enemy_completed_supply_depots),
            len(enemy_barrackses),
            #len(enemy_completed_barrackses),
            len(enemy_marines),
            len(my_unit_at_1_1),
            len(my_unit_at_1_2), 
            len(my_unit_at_1_3), 
            len(my_unit_at_2_1),
            len(my_unit_at_2_2),
            len(my_unit_at_2_3),
            len(my_unit_at_3_1),
            len(my_unit_at_3_2),
            len(my_unit_at_3_3),
            len(enemy_unit_at_1_1),
            len(enemy_unit_at_1_2),
            len(enemy_unit_at_1_3),
            len(enemy_unit_at_2_1),
            len(enemy_unit_at_2_2),
            len(enemy_unit_at_2_3),
            len(enemy_unit_at_3_1),
            len(enemy_unit_at_3_2),
            len(enemy_unit_at_3_3)
            )
    
  def step(self, obs):
      
    if obs.last():
      self.qtable.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')
      self.qtable.q_table.to_csv(DATA_FILE + '.csv')
    super(SmartAgent, self).step(obs)
    state = str(self.get_state(obs))
    action = self.qtable.choose_action(state)
    #print(action)

    total_value_units_score = obs.observation['score_cumulative'][3]
    total_value_structures_score = obs.observation['score_cumulative'][4]
    killed_value_units_score = obs.observation['score_cumulative'][5]
    killed_value_structures_score = obs.observation['score_cumulative'][6]
    self.previous_total_value_units = 0
    self.previous_total_value_structures = 0
    #print(obs.observation['score_cumulative'][11])
    total_spent_minerals = obs.observation['score_cumulative'][11]

    if self.previous_action is not None:
      step_reward = 0
      if total_value_units_score < self.previous_total_value_units_score:
        step_reward -= DEAD_UNIT_REWARD_RATE * (self.previous_total_value_units_score - total_value_units_score)

      if total_value_structures_score < self.previous_total_value_structures_score:
        step_reward -= DEAD_BUILDING_REWARD_RATE * (self.previous_total_value_structures_score - total_value_structures_score)

      if killed_value_units_score > self.previous_killed_value_units_score:
          step_reward += KILL_UNIT_REWARD_RATE * (killed_value_units_score - self.previous_killed_value_units_score)
              
      if killed_value_structures_score > self.previous_killed_value_structures_score:
          step_reward += KILL_BUILDING_REWARD_RATE * (killed_value_structures_score - self.previous_killed_value_structures_score)
      
      if total_spent_minerals > self.previous_total_spent_minerals:
          step_reward += MORE_MINERALS_USED_REWARD_RATE * (total_spent_minerals - self.previous_total_spent_minerals)
        
      self.qtable.learn(self.previous_state,
                        self.previous_action,
                        obs.reward + step_reward,
                        'terminal' if obs.last() else state)

    self.previous_total_value_units_score = total_value_units_score
    self.previous_total_value_structures_score = total_value_structures_score
    self.previous_killed_unit_score = killed_value_units_score
    self.previous_killed_value_structures_score = killed_value_structures_score
    self.previous_total_spent_minerals = total_spent_minerals
    self.previous_state = state
    self.previous_action = action
    return getattr(self, action)(obs)

def main(unused_argv):
  agent1 = SmartAgent()
  #agent2 = RandomAgent()
  try:
    with sc2_env.SC2Env(
        map_name="Simple64",
        players=[sc2_env.Agent(sc2_env.Race.terran), 
                 sc2_env.Bot(sc2_env.Race.terran, 
                               sc2_env.Difficulty.very_easy)],
                 #sc2_env.Agent(sc2_env.Race.terran)],
        agent_interface_format=features.AgentInterfaceFormat(
            action_space=actions.ActionSpace.RAW,
            use_raw_units=True,
            raw_resolution=64,
        ),
        step_mul=48,
        disable_fog=True,
    ) as env:
      #run_loop.run_loop([agent1, agent2], env, max_episodes=1000)
      run_loop.run_loop([agent1], env, max_episodes=1000)
  except KeyboardInterrupt:
    pass


if __name__ == "__main__":
  app.run(main)
  