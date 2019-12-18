import random
import numpy as np
import pandas as pd
import os
import logging

from absl import app
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env, run_loop

from base_agent import *
import sub_policy_battle
import sub_policy_economic
import sub_policy_training

os.close(2)

DATA_FILE = 'AI_agent_data'


class Agent(BaseAgent):

    battle_policy = sub_policy_battle.SubAgent_Battle()
    economic_policy = sub_policy_economic.SubAgent_Economic()
    training_policy = sub_policy_training.SubAgent_Training()

    actions = ("choose_battle_policy",
               "choose_economic_policy",
               "choose_training_policy",
               )

    def choose_battle_policy(self, obs):
        """
        call sub policy to choose a action
        Args:  observation
        Returns: action(string)
        """
        log.debug('in choose battle')
        choose_action = self.battle_policy.step(obs)
        log.debug('out choose battle')
        return choose_action

    def choose_economic_policy(self, obs):
        """
        call sub policy to choose a action
        Args:  observation
        Returns: action(string)
        """
        log.debug('in choose economic')
        choose_action = self.economic_policy.step(obs)
        log.debug('out choose economic')
        return choose_action

    def choose_training_policy(self, obs):
        """
        call sub policy to choose a action
        Args:  observation
        Returns: action(string)
        """
        log.debug('in choose training')
        choose_action = self.training_policy.step(obs)
        log.debug('out choose training')
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
        log.debug('in __init__')
        super(SmartAgent, self).__init__()
        self.qtable = QLearningTable(self.actions)
        if os.path.isfile(DATA_FILE + '.gz'):
            self.qtable.q_table = pd.read_pickle(
                DATA_FILE + '.gz', compression='gzip')
        self.new_game()

    def reset(self):
        log.debug('in reset')
        if self.episodes != 0:
            log.warning(
                f"Episode {self.episodes} finished after {self.steps} game steps. Score: {self.score}")
        super(SmartAgent, self).reset()
        log.warning(f"Starting episode {self.episodes}")
        self.new_game()
        self.battle_policy.reset()
        self.economic_policy.reset()
        self.training_policy.reset()

    def new_game(self):
        log.debug('in new game')
        self.base_top_left = None
        self.previous_state = None
        self.previous_action = None
        self.previous_total_value_units_score = 0
        self.previous_total_value_structures_score = 0
        self.previous_killed_value_units_score = 0
        self.previous_killed_value_structures_score = 0
        self.previous_total_spent_minerals = 0

        self.score = 0

    def get_state(self, obs):
        """
        get state of starcraft II
        Args:  observation
        Returns: state(list)
        """
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
        enemy_barrackses = self.get_enemy_units_by_type(
            obs, units.Terran.Barracks)
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
        """
        every step starcraft II will call this function
        return: getattr(self, action)(obs)
        """
        log.debug('into step')
        if obs.last():
            self.qtable.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')
            self.battle_policy.save_module()
            self.economic_policy.save_module()
            self.training_policy.save_module()
        super(SmartAgent, self).step(obs)
        state = str(self.get_state(obs))
        action = self.qtable.choose_action(state)
        log.info(action)

        if self.previous_action is not None:
            step_reward = 0
            self.qtable.learn(self.previous_state,
                              self.previous_action,
                              obs.reward + step_reward,
                              'terminal' if obs.last() else state)

        self.previous_state = state
        self.previous_action = action

        # record score for episode ending use
        self.score = obs.observation.score_cumulative.score
        log.debug('get out step')
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
            # sc2_env.Agent(sc2_env.Race.terran)],
            agent_interface_format=features.AgentInterfaceFormat(
                action_space=actions.ActionSpace.RAW,
                use_raw_units=True,
                raw_resolution=64,
            ),
            step_mul=48,
            disable_fog=False,
        ) as env:
            #run_loop.run_loop([agent1, agent2], env, max_episodes=1000)
            run_loop.run_loop([agent1], env, max_episodes=1000)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    app.run(main)
