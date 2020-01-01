from base_agent import *

from unit.units import Upgrade
import unit.terran_unit as terran
import unit.terran_upgrade as terran_upgrade

DATA_FILE = 'Sub_building_data'
FAILED_COMMAND = 0.0001
MORE_MINERALS_USED_REWARD_RATE = 0.0001
MORE_VESPENE_USED_REWARD_RATE = 0.0002

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 500

BUILD_RANGE_MAX = 6
BUILD_RANGE_MIN = 2

SAVE_POLICY_NET = 'model/economic_dqn_policy'
SAVE_TARGET_NET = 'model/economic_dqn_target'
SAVE_MEMORY = 'model/economic_memory'



class Agent(BaseAgent):

    actions = tuple(["do_nothing",
               "train_SCV",
               "harvest_minerals",
               "harvest_gas"] +\
                [f"build_{building}" for building in BUILDING_UNIT_NAME] +\
                [f"research_{tech}" for tech in RESEARCH_NAME]
               )

    def __init__(self):
        super(Agent, self).__init__()
        for tech in RESEARCH_NAME:
            self.__setattr__(f"research_{tech}", partial(self.research_tech, upgrade=tech))

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
                and unit.alliance == features.PlayerRelative.SELF
                and (unit.mineral_contents > 0 or unit.vespene_contents > 0)
                ]

    def harvest_minerals(self, obs):
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        idle_scvs = [scv for scv in scvs if scv.order_length == 0]
        if len(idle_scvs) > 0:
            mineral_patches = [unit for unit in obs.observation.raw_units
                               if unit.unit_type in [
                                   units.Neutral.MineralField,
                                   units.Neutral.MineralField750,
                               ]]
            scv = random.choice(idle_scvs)
            distances = self.get_distances(
                obs, mineral_patches, (scv.x, scv.y))
            mineral_patch = mineral_patches[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Harvest_Gather_unit(
                "now", [scv.tag for scv in idle_scvs], mineral_patch.tag)
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

    def check_if_buildable(self, obs, posx, posy):
        if self.get_my_building_by_pos(obs, posx, posy, posx+1, posy+1) == [] and \
           self.get_enemy_building_by_pos(obs, posx, posy, posx+1, posy+1) == []:
            return True
        return False
        
#build something
    def build_CommandCenter(self, obs):
        command_centers = self.get_my_units_by_type(obs, units.Terran.CommandCenter)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        if len(command_centers) == 0 and len(scvs) > 0:
            build_xy = (19, 23) if self.base_top_left else (39, 45)
            distances = self.get_distances(obs, scvs, build_xy)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_CommandCenter_pt("now", scv.tag, build_xy)
        if len(command_centers) == 1 and len(scvs) > 0:
            build_xy = (41, 21) if self.base_top_left else (17, 48)
            distances = self.get_distances(obs, scvs, build_xy)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_CommandCenter_pt("now", scv.tag, build_xy)
        elif len(command_centers) == 2 and len(scvs) > 0:
            build_xy = (17, 48) if self.base_top_left else (41, 21)
            distances = self.get_distances(obs, scvs, build_xy)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_CommandCenter_pt("now", scv.tag, build_xy)
        #(19, 23)(41, 21)(17, 48)(39, 45)
        return actions.RAW_FUNCTIONS.no_op()

    def build_Refinery(self, obs):
        refinery = self.get_my_units_by_type(obs, units.Terran.Refinery)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        gas_patches = [unit for unit in obs.observation.raw_units
                       if unit.unit_type == units.Neutral.VespeneGeyser ]
        if len(refinery) >= 0 and len(scvs) > 0:
            scv = random.choice(scvs)
            distances = self.get_distances(obs, gas_patches, (scv.x, scv.y))
            gas_patch = gas_patches[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_Refinery_pt("now", scv.tag, gas_patch.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def build_SupplyDepot(self, obs):
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        if len(scvs) > 0:
            build_xy = (MAIN_COMMAND_CENTER_POTISION[self.base_top_left]['x'] + random.randint(BUILD_RANGE_MIN, BUILD_RANGE_MAX) * random.choice([-1, 1]),
                        MAIN_COMMAND_CENTER_POTISION[self.base_top_left]['y'] + random.randint(BUILD_RANGE_MIN, BUILD_RANGE_MAX) * random.choice([-1, 1]))                
            for i in range(0, 10):
                if not self.check_if_buildable(obs, *build_xy):
                    build_xy = (MAIN_COMMAND_CENTER_POTISION[self.base_top_left]['x'] + random.randint(BUILD_RANGE_MIN, BUILD_RANGE_MAX) * random.choice([-1, 1]),
                                MAIN_COMMAND_CENTER_POTISION[self.base_top_left]['y'] + random.randint(BUILD_RANGE_MIN, BUILD_RANGE_MAX) * random.choice([-1, 1]))     
            distances = self.get_distances(obs, scvs, build_xy)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt(
                "now", scv.tag, build_xy)
        return actions.RAW_FUNCTIONS.no_op()

    def build_Barracks(self, obs):
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        if (len(scvs) > 0):
            build_xy = (MAIN_COMMAND_CENTER_POTISION[self.base_top_left]['x'] + random.randint(BUILD_RANGE_MIN, BUILD_RANGE_MAX) * random.choice([-1, 1]),
                        MAIN_COMMAND_CENTER_POTISION[self.base_top_left]['y'] + random.randint(BUILD_RANGE_MIN, BUILD_RANGE_MAX) * random.choice([-1, 1]))                
            for i in range(0, 10):
                if not self.check_if_buildable(obs, *build_xy):
                    build_xy = (MAIN_COMMAND_CENTER_POTISION[self.base_top_left]['x'] + random.randint(BUILD_RANGE_MIN, BUILD_RANGE_MAX) * random.choice([-1, 1]),
                                MAIN_COMMAND_CENTER_POTISION[self.base_top_left]['y'] + random.randint(BUILD_RANGE_MIN, BUILD_RANGE_MAX) * random.choice([-1, 1]))     
            distances = self.get_distances(obs, scvs, build_xy)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_Barracks_pt(
                "now", scv.tag, build_xy)
        return actions.RAW_FUNCTIONS.no_op()

    def build_BarracksTechLab(self, obs):
        completed_barrackses = self.get_my_completed_units_by_type(
            obs, units.Terran.Barracks)
        if len(completed_barrackses) > 0:
            return actions.RAW_FUNCTIONS.Build_TechLab_Barracks_quick(
                "now", [barrack.tag for barrack in completed_barrackses])
        return actions.RAW_FUNCTIONS.no_op()

    def build_BarracksReactor(self, obs):
        completed_barrackses = self.get_my_completed_units_by_type(
            obs, units.Terran.Barracks)
        if len(completed_barrackses) > 0:
            return actions.RAW_FUNCTIONS.Build_Reactor_Barracks_quick(
                "now", [barrack.tag for barrack in completed_barrackses])
        return actions.RAW_FUNCTIONS.no_op()

    def build_GhostAcademy(self, obs):
        ghost_academy_list = self.get_my_units_by_type(obs, units.Terran.GhostAcademy)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        if (len(scvs) > 0 and len(ghost_academy_list) == 0):
            build_xy = (MAIN_COMMAND_CENTER_POTISION[self.base_top_left]['x'] + random.randint(BUILD_RANGE_MIN, BUILD_RANGE_MAX) * random.choice([-1, 1]),
                        MAIN_COMMAND_CENTER_POTISION[self.base_top_left]['y'] + random.randint(BUILD_RANGE_MIN, BUILD_RANGE_MAX) * random.choice([-1, 1]))                
            for i in range(0, 10):
                if not self.check_if_buildable(obs, *build_xy):
                    build_xy = (MAIN_COMMAND_CENTER_POTISION[self.base_top_left]['x'] + random.randint(BUILD_RANGE_MIN, BUILD_RANGE_MAX) * random.choice([-1, 1]),
                                MAIN_COMMAND_CENTER_POTISION[self.base_top_left]['y'] + random.randint(BUILD_RANGE_MIN, BUILD_RANGE_MAX) * random.choice([-1, 1]))     
            distances = self.get_distances(obs, scvs, build_xy)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_GhostAcademy_pt(
                "now", scv.tag, build_xy)
        return actions.RAW_FUNCTIONS.no_op()

    def build_EngineeringBay(self, obs):
        engineering_bay_list = self.get_my_units_by_type(obs, units.Terran.EngineeringBay)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        if (len(scvs) > 0 and len(engineering_bay_list) == 0):
            build_xy = (MAIN_COMMAND_CENTER_POTISION[self.base_top_left]['x'] + random.randint(BUILD_RANGE_MIN, BUILD_RANGE_MAX) * random.choice([-1, 1]),
                        MAIN_COMMAND_CENTER_POTISION[self.base_top_left]['y'] + random.randint(BUILD_RANGE_MIN, BUILD_RANGE_MAX) * random.choice([-1, 1]))                
            for i in range(0, 10):
                if not self.check_if_buildable(obs, *build_xy):
                    build_xy = (MAIN_COMMAND_CENTER_POTISION[self.base_top_left]['x'] + random.randint(BUILD_RANGE_MIN, BUILD_RANGE_MAX) * random.choice([-1, 1]),
                                MAIN_COMMAND_CENTER_POTISION[self.base_top_left]['y'] + random.randint(BUILD_RANGE_MIN, BUILD_RANGE_MAX) * random.choice([-1, 1]))     
            distances = self.get_distances(obs, scvs, build_xy)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_EngineeringBay_pt(
                "now", scv.tag, build_xy)
        return actions.RAW_FUNCTIONS.no_op()

    def build_Factory(self, obs):
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        if (len(scvs) > 0):
            build_xy = (MAIN_COMMAND_CENTER_POTISION[self.base_top_left]['x'] + random.randint(BUILD_RANGE_MIN, BUILD_RANGE_MAX) * random.choice([-1, 1]),
                        MAIN_COMMAND_CENTER_POTISION[self.base_top_left]['y'] + random.randint(BUILD_RANGE_MIN, BUILD_RANGE_MAX) * random.choice([-1, 1]))                
            for i in range(0, 10):
                if not self.check_if_buildable(obs, *build_xy):
                    build_xy = (MAIN_COMMAND_CENTER_POTISION[self.base_top_left]['x'] + random.randint(BUILD_RANGE_MIN, BUILD_RANGE_MAX) * random.choice([-1, 1]),
                                MAIN_COMMAND_CENTER_POTISION[self.base_top_left]['y'] + random.randint(BUILD_RANGE_MIN, BUILD_RANGE_MAX) * random.choice([-1, 1]))     
            distances = self.get_distances(obs, scvs, build_xy)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_Factory_pt(
                "now", scv.tag, build_xy)
        return actions.RAW_FUNCTIONS.no_op()

    def build_FactoryTechLab(self, obs):
        completed_factorys = self.get_my_completed_units_by_type(
            obs, units.Terran.Factory)
        if len(completed_factorys) > 0:
            return actions.RAW_FUNCTIONS.Build_TechLab_Factory_quick(
                "now", [factorys.tag for factorys in completed_factorys])
        return actions.RAW_FUNCTIONS.no_op()

    def build_FactoryReactor(self, obs):
        completed_factorys = self.get_my_completed_units_by_type(
            obs, units.Terran.Factory)
        if len(completed_factorys) > 0:
            return actions.RAW_FUNCTIONS.Build_Reactor_Factory_quick(
                "now", [factorys.tag for factorys in completed_factorys])
        return actions.RAW_FUNCTIONS.no_op()

    def build_Armory(self, obs):
        armory_list = self.get_my_units_by_type(obs, units.Terran.Armory)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        if (len(scvs) > 0 and len(armory_list) == 0):
            build_xy = (MAIN_COMMAND_CENTER_POTISION[self.base_top_left]['x'] + random.randint(BUILD_RANGE_MIN, BUILD_RANGE_MAX) * random.choice([-1, 1]),
                        MAIN_COMMAND_CENTER_POTISION[self.base_top_left]['y'] + random.randint(BUILD_RANGE_MIN, BUILD_RANGE_MAX) * random.choice([-1, 1]))                
            for i in range(0, 10):
                if not self.check_if_buildable(obs, *build_xy):
                    build_xy = (MAIN_COMMAND_CENTER_POTISION[self.base_top_left]['x'] + random.randint(BUILD_RANGE_MIN, BUILD_RANGE_MAX) * random.choice([-1, 1]),
                                MAIN_COMMAND_CENTER_POTISION[self.base_top_left]['y'] + random.randint(BUILD_RANGE_MIN, BUILD_RANGE_MAX) * random.choice([-1, 1]))     
            distances = self.get_distances(obs, scvs, build_xy)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_Armory_pt(
                "now", scv.tag, build_xy)
        return actions.RAW_FUNCTIONS.no_op()

    def build_Starport(self, obs):
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        if (len(scvs) > 0):
            build_xy = (MAIN_COMMAND_CENTER_POTISION[self.base_top_left]['x'] + random.randint(BUILD_RANGE_MIN, BUILD_RANGE_MAX) * random.choice([-1, 1]),
                        MAIN_COMMAND_CENTER_POTISION[self.base_top_left]['y'] + random.randint(BUILD_RANGE_MIN, BUILD_RANGE_MAX) * random.choice([-1, 1]))                
            for i in range(0, 10):
                if not self.check_if_buildable(obs, *build_xy):
                    build_xy = (MAIN_COMMAND_CENTER_POTISION[self.base_top_left]['x'] + random.randint(BUILD_RANGE_MIN, BUILD_RANGE_MAX) * random.choice([-1, 1]),
                                MAIN_COMMAND_CENTER_POTISION[self.base_top_left]['y'] + random.randint(BUILD_RANGE_MIN, BUILD_RANGE_MAX) * random.choice([-1, 1]))     
            distances = self.get_distances(obs, scvs, build_xy)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_Starport_pt(
                "now", scv.tag, build_xy)
        return actions.RAW_FUNCTIONS.no_op()

    def build_FusionCore(self, obs):
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        if (len(scvs) > 0):
            build_xy = (MAIN_COMMAND_CENTER_POTISION[self.base_top_left]['x'] + random.randint(BUILD_RANGE_MIN, BUILD_RANGE_MAX) * random.choice([-1, 1]),
                        MAIN_COMMAND_CENTER_POTISION[self.base_top_left]['y'] + random.randint(BUILD_RANGE_MIN, BUILD_RANGE_MAX) * random.choice([-1, 1]))                
            for i in range(0, 10):
                if not self.check_if_buildable(obs, *build_xy):
                    build_xy = (MAIN_COMMAND_CENTER_POTISION[self.base_top_left]['x'] + random.randint(BUILD_RANGE_MIN, BUILD_RANGE_MAX) * random.choice([-1, 1]),
                                MAIN_COMMAND_CENTER_POTISION[self.base_top_left]['y'] + random.randint(BUILD_RANGE_MIN, BUILD_RANGE_MAX) * random.choice([-1, 1]))     
            distances = self.get_distances(obs, scvs, build_xy)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_FusionCore_pt(
                "now", scv.tag, build_xy)
        return actions.RAW_FUNCTIONS.no_op()

    def build_StarportTechLab(self, obs):
        completed_starports = self.get_my_completed_units_by_type(
            obs, units.Terran.Starport)
        if len(completed_starports) > 0:
            return actions.RAW_FUNCTIONS.Build_TechLab_Starport_quick(
                "now", [starports.tag for starports in completed_starports])
        return actions.RAW_FUNCTIONS.no_op()

    def build_StarportReactor(self, obs):
        completed_starports = self.get_my_completed_units_by_type(
            obs, units.Terran.Starport)
        if len(completed_starports) > 0:
            return actions.RAW_FUNCTIONS.Build_Reactor_Starport_quick(
                "now", [starports.tag for starports in completed_starports])
        return actions.RAW_FUNCTIONS.no_op()

#Research all
    def research_tech(self, obs, upgrade=None):
        """ research each technology
        Args:
            obs
            upgrade (string): class name in unit.terran_upgrade
        """
        tech = getattr(terran_upgrade, upgrade)
        building_to_research = self.get_my_completed_units_by_type(obs, getattr(terran, tech.research_from)().index)
        if len(building_to_research) > 0:
            if tech.upgrade() == True:
                return actions.RAW_FUNCTIONS.__getattr__(f"Research_{upgrade}_quick")("now", [b.tag for b in building_to_research])
        return actions.RAW_FUNCTIONS.no_op()


class SubAgent_Economic(Agent):

    def __init__(self):
        super(SubAgent_Economic, self).__init__()
        self.new_game()
        self.set_DQN(SAVE_POLICY_NET, SAVE_TARGET_NET, SAVE_MEMORY)
        self.episode = 0

    def reset(self):
        log.debug('in reset') 
        super(SubAgent_Economic, self).reset()
        
        ## reset upgrade
        for upgrade_class in Upgrade.already_upgrade:
            upgrade_class.reset()
        
        self.new_game()

    def new_game(self):
        log.debug('in new game')
        self.base_top_left = None
        self.previous_state = None
        self.previous_action = None
        self.previous_total_value_units_score = 0
        self.previous_total_value_structures_score = 0
        self.previous_total_spent_minerals = 0
        self.previous_total_spent_vespene = 0
        self.negative_reward = 0

    def can_afford_unit(self, obs, action):
        """ Check if all conditi on can afford to this unit
        Args:
          obs
          action (str): from self.actions
        Returns:
          True: if can afford
          or                                                                
          False: if can not afford
        """
        can_afford = False
        
        # Parse action
        ## build_xxx
        if action.find("build") != -1:
            building = getattr(terran, action[action.index("_")+1:])()

            ### check requirement
            if len(self.get_my_units_by_type(obs, units.Terran.SCV)) > 0 \
            and obs.observation.player.minerals >= building.mineral_price\
            and obs.observation.player.vespene >= building.vespene_price \
            and 0 not in [len(self.get_my_completed_units_by_type(obs, getattr(terran, requirement)().index)) for requirement in building.requirements]:
                can_afford = True
            
            ### Reactor or TechLab
            if isinstance(building, terran.Reactor) or isinstance(building, terran.TechLab):
                father_building = self.get_my_units_by_type(obs, getattr(terran, building.requirements[0])().index)
                sibling_building = \
                        self.get_my_units_by_type(obs, getattr(terran, building.requirements[0]+"Reactor")().index) + \
                        self.get_my_units_by_type(obs, getattr(terran, building.requirements[0]+"TechLab")().index)

                if len(father_building) <= len(sibling_building):
                    can_afford = False

        ## research_xxx
        elif action.find("research") != -1:
            tech = getattr(terran_upgrade, action[action.index("_")+1:])
            
            ### check requirement
            if len(self.get_my_completed_units_by_type(obs, getattr(terran, tech.research_from)().index)) > 0 \
            and obs.observation.player.minerals >= tech.mineral_price\
            and obs.observation.player.vespene >= tech.vespene_price \
            and 0 not in [len(self.get_my_completed_units_by_type(obs, getattr(terran, requirement)().index)) for requirement in tech.requirements]:
                can_afford = True
            
        return can_afford

    def get_negative_reward(self, obs, action):

        player_mineral = obs.observation.player.minerals
        player_vespene = obs.observation.player.vespene
        free_supply = (obs.observation.player.food_cap - obs.observation.player.food_used)

        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        idle_scvs = [scv for scv in scvs if scv.order_length == 0]


        if self.can_afford_unit(obs, action) == False:
            return FAILED_COMMAND

        if action == "harvest_minerals":
            if len(idle_scvs) == 0:
                return FAILED_COMMAND
        elif action == "train_SCV":
            completed_commandcenters = self.get_my_completed_units_by_type(obs, units.Terran.CommandCenter)
            if len(completed_commandcenters) <= 0 or free_supply == 0 or player_mineral < 50:
                return FAILED_COMMAND
        elif action == "harvest_gas":
            notfull_completed_refinerys = self.get_notfull_worker_building_by_type(obs, units.Terran.Refinery)
            if len(scvs) <= 0 or len(notfull_completed_refinerys) <= 0:
                return FAILED_COMMAND
        return 0

    def get_state(self, obs):

        player_mineral = obs.observation.player.minerals
        player_vespene = obs.observation.player.vespene

        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        idle_scvs = [scv for scv in scvs if scv.order_length == 0]
        buildings = [len(self.get_my_completed_units_by_type(obs, getattr(units.Terran, unit))) for unit in BUILDING_UNIT_NAME] 


        player_food_used = obs.observation.player.food_used
        player_food_cap = obs.observation.player.food_cap
        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)

        return tuple([self.base_top_left,
                player_mineral,
                player_vespene,
                len(scvs),
                len(idle_scvs)]+ \
                buildings + \
                [free_supply,
                player_food_used,
                player_food_cap]
                )

    def step(self, obs):
        super(SubAgent_Economic, self).step(obs)
            
        self.episode += 1
        state = self.get_state(obs)
        log.debug(f"state: {state}")
        action, action_idx = self.select_action(state)
        log.info(action)

        #a_com = self.get_my_units_by_type(obs, units.Terran.CommandCenter)
        #print([a_com[0].x, a_com[0].y])

        if self.episodes % 3 == 2:
            if self.previous_action is not None:
                step_reward = self.get_reward(obs)

                log.log(LOG_REWARD, "economic reward = " + str(obs.reward + step_reward))
                if not obs.last():
                    self.memory.push(torch.Tensor(self.previous_state).to(device),
                                    torch.LongTensor([self.previous_action_idx]).to(device),
                                    torch.Tensor(state).to(device),
                                    torch.Tensor([obs.reward + step_reward]).to(device))
                    self.optimize_model()
                else: # last step in an episode
                    pass
            else: # first step
                pass

        if self.episode % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.previous_state = state
        self.previous_action = action
        self.previous_action_idx = action_idx
        return getattr(self, action)(obs)

    def get_reward(self, obs):
        total_value_units_score = obs.observation.score_cumulative.total_value_units
        total_value_structures_score = obs.observation.score_cumulative.total_value_structures
        total_spent_minerals = obs.observation.score_cumulative.spent_minerals
        total_spent_vespene = obs.observation.score_cumulative.spent_vespene

        positive_reward = 0
        ## Positive reward update in this step

        ## Negative reward update in next step

        if total_spent_minerals > self.previous_total_spent_minerals:
            positive_reward += MORE_MINERALS_USED_REWARD_RATE * \
                (total_spent_minerals - self.previous_total_spent_minerals)
        if total_spent_vespene > self.previous_total_spent_vespene:
            positive_reward += MORE_VESPENE_USED_REWARD_RATE * \
                (total_spent_vespene - self.previous_total_spent_vespene)

        step_reward = positive_reward - self.negative_reward
        self.negative_reward = self.get_negative_reward(obs, self.previous_action)

        self.previous_total_value_units_score = total_value_units_score
        self.previous_total_value_structures_score = total_value_structures_score
        self.previous_total_spent_minerals = total_spent_minerals
        self.previous_total_spent_vespene = total_spent_vespene
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
                _, idx = self.policy_net(torch.Tensor(state).to(device)).max(0)
                return self.actions[idx], idx
        else:
            idx = random.randrange(self.action_size)
            return self.actions[idx], idx

    def save_module(self):
        self.policy_net.save()
        self.target_net.save()
        with open(SAVE_MEMORY, 'wb') as f:
            pickle.dump(self.memory, f)
            log.log(LOG_MODEL, "Save memory in economic")
