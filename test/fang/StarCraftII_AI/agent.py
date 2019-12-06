import random
import numpy as np
import pandas as pd
import os
from absl import app
from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env, run_loop
import sub_policy_battle
import sub_policy_economic
import sub_policy_training


DATA_FILE = 'AI_agent_data'

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

  battle_policy = sub_policy_battle.SubAgent_Battle()
  economic_policy = sub_policy_economic.SubAgent_Economic()
  training_policy = sub_policy_training.SubAgent_Training()

  actions = ("choose_battle_policy",
             "choose_economic_policy",
             "choose_training_policy",
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
              

  def choose_battle_policy(self, obs):
    #print('in choose battle')
    choose_action = self.battle_policy.step(obs)
    #print('out choose battle')
    print(choose_action)
    return choose_action

  def choose_economic_policy(self, obs):
    #print('in choose economic')
    choose_action = self.economic_policy.step(obs)
    #print('out choose economic')
    print(choose_action)
    return choose_action

  def choose_training_policy(self, obs):
    #print('in choose training')
    choose_action = self.training_policy.step(obs)
    #print('out choose training')
    print(choose_action)
    return choose_action

  def step(self, obs):
    super(Agent, self).step(obs)
    if obs.first():
      command_center = self.get_my_units_by_type(
          obs, units.Terran.CommandCenter)[0]
      self.base_top_left = (command_center.x < 32)
      self.battle_policy.set_top_left(obs)
      self.economic_policy.set_top_left(obs)
      self.training_policy.set_top_left(obs)

      
class RandomAgent(Agent):
  def step(self, obs):
    super(RandomAgent, self).step(obs)
    action = random.choice(self.actions)
    return getattr(self, action)(obs)

class SmartAgent(Agent):

  def __init__(self):
    #print('in __init__')
    super(SmartAgent, self).__init__()
    self.qtable = QLearningTable(self.actions)
    if os.path.isfile(DATA_FILE + '.gz'):
      self.qtable.q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')
    self.new_game()

    
  def reset(self):
    #print('in reset')
    super(SmartAgent, self).reset()
    self.new_game()
    self.battle_policy.reset()
    self.economic_policy.reset()
    self.training_policy.reset()
    
  def new_game(self):
    #print('in new game')
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

    marines = self.get_my_units_by_type(obs, units.Terran.Marine) 
    
    free_supply = (obs.observation.player.food_cap - 
                   obs.observation.player.food_used)
    can_afford_supply_depot = obs.observation.player.minerals >= 100
    can_afford_barracks = obs.observation.player.minerals >= 150
    can_afford_marine = obs.observation.player.minerals >= 100
    
    enemy_scvs = self.get_enemy_units_by_type(obs, units.Terran.SCV)
    enemy_command_centers = self.get_enemy_units_by_type(
        obs, units.Terran.CommandCenter)
    enemy_supply_depots = self.get_enemy_units_by_type(
        obs, units.Terran.SupplyDepot)
    enemy_barrackses = self.get_enemy_units_by_type(obs, units.Terran.Barracks)
    enemy_marines = self.get_enemy_units_by_type(obs, units.Terran.Marine)
    enemy_unit_num = self.get_enemy_units_by_pos(obs, 0, 0, 64, 64)
    

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
            can_afford_barracks,
            can_afford_marine,
            len(enemy_command_centers),
            len(enemy_scvs),
            len(enemy_supply_depots),
            len(enemy_barrackses),
            len(enemy_marines),
            len(enemy_unit_num)
            )
    
  def step(self, obs):
    #print('into step')
    if obs.last():
      self.qtable.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')
      self.qtable.q_table.to_csv(DATA_FILE + '.csv')
      self.battle_policy.save_module()
      self.economic_policy.save_module()
      self.training_policy.save_module()
    super(SmartAgent, self).step(obs)
    state = str(self.get_state(obs))
    action = self.qtable.choose_action(state)
    print(action)

    if self.previous_action is not None:
      step_reward = 0
      self.qtable.learn(self.previous_state,
                        self.previous_action,
                        obs.reward + step_reward,
                        'terminal' if obs.last() else state)

    self.previous_state = state
    self.previous_action = action
    #print('get out step')
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