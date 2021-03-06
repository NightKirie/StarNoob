import random
import numpy as np
import pandas as pd
import os
from absl import app
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env, run_loop

from base_agent import QLearningTable
import base_agent

DATA_FILE = 'Sub_building_data'
MORE_MINERALS_USED_REWARD_RATE = 0.00001


class Agent(base_agent.BaseAgent):

    actions = ("do_nothing",
               "harvest_minerals",
               "build_supply_depot",
               "build_barracks",
               "build_refinery",
               "train_SCV",
               "harvest_gas",
               "build_barrack_techlab",
               "build_barrack_reactor",
               "build_engineeringbays",
               "research_infantryweapons",
               "research_infantryarmor",
               "research_hiSecautotracking",
               "research_structurearmor",
               )

    def get_distances(self, obs, units, xy):
        units_xy = [(unit.x, unit.y) for unit in units]
        return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1)

    def get_least_busy_building(self, building):
        order_array = []
        for item in building:
            order_array.append(item.order_length)
        return order_array.index(min(order_array))

    def get_notfull_worker_building_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.build_progress == 100
                and unit.assigned_harvesters < unit.ideal_harvesters
                and unit.alliance == features.PlayerRelative.SELF]

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
            distances = self.get_distances(
                obs, mineral_patches, (scv.x, scv.y))
            mineral_patch = mineral_patches[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Harvest_Gather_unit(
                "now", scv.tag, mineral_patch.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def harvest_gas(self, obs):
        notfull_completed_refinerys = self.get_notfull_worker_building_by_type(
            obs, units.Terran.Refinery)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        if len(scvs) > 0 and len(notfull_completed_refinerys) > 0:
            scv = random.choice(scvs)
            distances = self.get_distances(
                obs, notfull_completed_refinerys, (scv.x, scv.y))
            refinery = notfull_completed_refinerys[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Harvest_Gather_unit(
                "now", scv.tag, refinery.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def train_SCV(self, obs):
        completed_commandcenters = self.get_my_completed_units_by_type(
            obs, units.Terran.CommandCenter)
        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)

        if len(completed_commandcenters) > 0 and free_supply > 0:
            completed_commandcenter = completed_commandcenters[self.get_least_busy_building(
                completed_commandcenters)]
            return actions.RAW_FUNCTIONS.Train_SCV_quick("now", completed_commandcenter.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def build_supply_depot(self, obs):
        supply_depots = self.get_my_units_by_type(
            obs, units.Terran.SupplyDepot)
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
        if (len(supply_depots) == 3 and obs.observation.player.minerals >= 100 and
                len(scvs) > 0):
            supply_depot_xy = (19, 26) if self.base_top_left else (38, 42)
            distances = self.get_distances(obs, scvs, supply_depot_xy)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt(
                "now", scv.tag, supply_depot_xy)

        return actions.RAW_FUNCTIONS.no_op()

    def build_refinery(self, obs):
        refinery = self.get_my_units_by_type(obs, units.Terran.Refinery)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        gas_patches = [unit for unit in obs.observation.raw_units
                       if unit.unit_type in [
                           units.Neutral.ProtossVespeneGeyser,
                           units.Neutral.PurifierVespeneGeyser,
                           units.Neutral.RichVespeneGeyser,
                           units.Neutral.ShakurasVespeneGeyser,
                           units.Neutral.SpacePlatformGeyser,
                           units.Neutral.VespeneGeyser
                       ]]
        if (len(refinery) == 0 and len(scvs) > 0):
            scv = random.choice(scvs)
            distances = self.get_distances(obs, gas_patches, (scv.x, scv.y))
            gas_patch = gas_patches[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_Refinery_pt("now", scv.tag, gas_patch.tag)
        if (len(refinery) == 1 and len(scvs) > 0):
            scv = random.choice(scvs)
            distances = self.get_distances(obs, gas_patches, (scv.x, scv.y))
            gas_patch = gas_patches[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_Refinery_pt("now", scv.tag, gas_patch.tag)
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
            barracks_xy = (22, 24) if self.base_top_left else (35, 47)
            distances = self.get_distances(obs, scvs, barracks_xy)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_Barracks_pt(
                "now", scv.tag, barracks_xy)
        return actions.RAW_FUNCTIONS.no_op()

    def build_barrack_techlab(self, obs):
        completed_barrackses = self.get_my_completed_units_by_type(
            obs, units.Terran.Barracks)
        if len(completed_barrackses) > 0:
            return actions.RAW_FUNCTIONS.Build_TechLab_Barracks_quick(
                "now", [barrack.tag for barrack in completed_barrackses])
        return actions.RAW_FUNCTIONS.no_op()

    def build_barrack_reactor(self, obs):
        completed_barrackses = self.get_my_completed_units_by_type(
            obs, units.Terran.Barracks)
        if len(completed_barrackses) > 0:
            return actions.RAW_FUNCTIONS.Build_Reactor_Barracks_quick(
                "now", [barrack.tag for barrack in completed_barrackses])
        return actions.RAW_FUNCTIONS.no_op()

    def build_engineeringbays(self, obs):
        engineeringbays = self.get_my_units_by_type(
            obs, units.Terran.EngineeringBay)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        if len(engineeringbays) == 0 and len(scvs) > 0:
            engineeringbays_xy = (19, 28) if self.base_top_left else (38, 40)
            distances = self.get_distances(obs, scvs, engineeringbays_xy)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_EngineeringBay_pt(
                "now", scv.tag, engineeringbays_xy)
        return actions.RAW_FUNCTIONS.no_op()

    def research_infantryweapons(self, obs):
        completed_engineeringbays = self.get_my_completed_units_by_type(
            obs, units.Terran.EngineeringBay)
        if len(completed_engineeringbays) > 0:
            return actions.RAW_FUNCTIONS.Research_TerranInfantryWeapons_quick(
                "now", [engineeringbay.tag for engineeringbay in completed_engineeringbays])
        return actions.RAW_FUNCTIONS.no_op()

    def research_infantryarmor(self, obs):
        completed_engineeringbays = self.get_my_completed_units_by_type(
            obs, units.Terran.EngineeringBay)
        if len(completed_engineeringbays) > 0:
            return actions.RAW_FUNCTIONS.Research_TerranInfantryArmor_quick(
                "now", [engineeringbay.tag for engineeringbay in completed_engineeringbays])
        return actions.RAW_FUNCTIONS.no_op()

    def research_hiSecautotracking(self, obs):
        completed_engineeringbays = self.get_my_completed_units_by_type(
            obs, units.Terran.EngineeringBay)
        if len(completed_engineeringbays) > 0:
            return actions.RAW_FUNCTIONS.Research_HiSecAutoTracking_quick(
                "now", [engineeringbay.tag for engineeringbay in completed_engineeringbays])
        return actions.RAW_FUNCTIONS.no_op()

    def research_structurearmor(self, obs):
        completed_engineeringbays = self.get_my_completed_units_by_type(
            obs, units.Terran.EngineeringBay)
        if len(completed_engineeringbays) > 0:
            return actions.RAW_FUNCTIONS.Research_TerranStructureArmorUpgrade_quick(
                "now", [engineeringbay.tag for engineeringbay in completed_engineeringbays])
        return actions.RAW_FUNCTIONS.no_op()


class SubAgent_Economic(Agent):

    def __init__(self):
        #print('in __init__')
        super(SubAgent_Economic, self).__init__()
        self.qtable = QLearningTable(self.actions)
        if os.path.isfile(DATA_FILE + '.gz'):
            self.qtable.q_table = pd.read_pickle(
                DATA_FILE + '.gz', compression='gzip')
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
        command_centers = self.get_my_units_by_type(
            obs, units.Terran.CommandCenter)

        refinerys = self.get_my_units_by_type(obs, units.Terran.Refinery)
        completed_refinerys = self.get_my_completed_units_by_type(
            obs, units.Terran.Refinery)

        supply_depots = self.get_my_units_by_type(
            obs, units.Terran.SupplyDepot)
        completed_supply_depots = self.get_my_completed_units_by_type(
            obs, units.Terran.SupplyDepot)

        barrackses = self.get_my_units_by_type(obs, units.Terran.Barracks)
        completed_barrackses = self.get_my_completed_units_by_type(
            obs, units.Terran.Barracks)

        techlabs = self.get_my_units_by_type(obs, units.Terran.BarracksTechLab)
        completed_techlabs = self.get_my_completed_units_by_type(
            obs, units.Terran.BarracksTechLab)

        reactors = self.get_my_units_by_type(obs, units.Terran.BarracksReactor)
        completed_reactors = self.get_my_completed_units_by_type(
            obs, units.Terran.BarracksReactor)

        engineeringbays = self.get_my_units_by_type(
            obs, units.Terran.EngineeringBay)
        completed_engineeringbays = self.get_my_completed_units_by_type(
            obs, units.Terran.EngineeringBay)

        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)

        can_afford_SCV = obs.observation.player.minerals >= 50
        can_afford_supply_depot = obs.observation.player.minerals >= 100
        can_afford_refinery = obs.observation.player.minerals >= 75
        can_afford_barracks = obs.observation.player.minerals >= 150
        can_afford_reactor = obs.observation.player.minerals >= 50 and obs.observation.player.vespene >= 50
        can_afford_techlab = obs.observation.player.minerals >= 50 and obs.observation.player.vespene >= 25
        can_afford_engineeringbays = obs.observation.player.minerals >= 125

        have_research_infantryweapons_level1 = 7 in obs.observation.upgrades
        have_research_infantryweapons_level2 = 8 in obs.observation.upgrades
        have_research_infantryweapons_level3 = 9 in obs.observation.upgrades
        have_research_infantryarmor_level1 = 11 in obs.observation.upgrades
        have_research_infantryarmor_level2 = 12 in obs.observation.upgrades
        have_research_infantryarmor_level3 = 13 in obs.observation.upgrades
        have_research_hiSecautotracking = 5 in obs.observation.upgrades
        have_research_structurearmor = 6 in obs.observation.upgrades
        #print(actions.RAW_FUNCTIONS.Research_HiSecAutoTracking_quick.id in actions.RAW_FUNCTIONS_AVAILABLE)
        print(obs.observation.upgrades)
        #print([have_research_hiSecautotracking, have_research_infantryweapons_level1, have_research_infantryweapons_level2, have_research_infantryweapons_level3])
        """
    if actions.RAW_FUNCTIONS.Build_EngineeringBay_pt.id in obs.observation['available_actions']:
      print('1')
    else:
      print('0')
    """

        return (self.base_top_left,
                len(command_centers),
                len(scvs),
                len(idle_scvs),
                len(refinerys),
                len(completed_refinerys),
                len(supply_depots),
                len(completed_supply_depots),
                len(barrackses),
                len(completed_barrackses),
                len(techlabs),
                len(completed_techlabs),
                len(reactors),
                len(completed_reactors),
                len(engineeringbays),
                len(completed_engineeringbays),
                free_supply,
                can_afford_SCV,
                can_afford_supply_depot,
                can_afford_refinery,
                can_afford_barracks,
                can_afford_reactor,
                can_afford_techlab,
                can_afford_engineeringbays,
                have_research_infantryweapons_level1,
                have_research_infantryweapons_level2,
                have_research_infantryweapons_level3,
                have_research_infantryarmor_level1,
                have_research_infantryarmor_level2,
                have_research_infantryarmor_level3,
                have_research_hiSecautotracking,
                have_research_structurearmor
                )

    def step(self, obs):

        if obs.last():
            self.qtable.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')
        super(SubAgent_Economic, self).step(obs)
        state = str(self.get_state(obs))
        action = self.qtable.choose_action(state)
        # print(action)

        total_value_units_score = obs.observation['score_cumulative'][3]
        total_value_structures_score = obs.observation['score_cumulative'][4]
        total_spent_minerals = obs.observation['score_cumulative'][11]

        if self.previous_action is not None:
            step_reward = 0
            # if total_value_units_score < self.previous_total_value_units_score:
            #  step_reward -= DEAD_UNIT_REWARD_RATE * (self.previous_total_value_units_score - total_value_units_score)

            # if total_value_structures_score < self.previous_total_value_structures_score:
            #  step_reward -= DEAD_BUILDING_REWARD_RATE * (self.previous_total_value_structures_score - total_value_structures_score)

            if total_spent_minerals > self.previous_total_spent_minerals:
                step_reward += MORE_MINERALS_USED_REWARD_RATE * \
                    (total_spent_minerals - self.previous_total_spent_minerals)

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
