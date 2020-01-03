from base_agent import *

from unit.units import Upgrade
import unit.terran_unit as terran
import unit.terran_upgrade as terran_upgrade

DATA_FILE = 'Sub_building_data'
FAILED_COMMAND = 0.0001
FAILED_BUILDING = 0.0001
MORE_MINERALS_USED_REWARD_RATE = 0.0001
MORE_VESPENE_USED_REWARD_RATE = 0.0002
TOO_MUCH_MINERAL_PENALTY = 0.0005
TOO_MUCH_VESPENE_PENALTY = 0.001

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
                [f"build_{building}" for building in MY_BUILDING_LIST] +\
                [f"research_{tech}" for tech in RESEARCH_NAME]
               )

    def __init__(self):
        super(Agent, self).__init__()
        for tech in RESEARCH_NAME:
            self.__setattr__(f"research_{tech}", partial(self.research_tech, upgrade=tech))
        
        for building in MY_BUILDING_LIST:
            self.__setattr__(f"build_{building}", partial(self.build_unit, building=building))

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
        mineral_patches = [unit for unit in obs.observation.raw_units
                               if unit.unit_type in [
                                   units.Neutral.MineralField,
                                   units.Neutral.MineralField750,
                               ]]
        if len(idle_scvs) > 0 and len(mineral_patches) > 0:
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
    
    def get_proper_position_to_build(self, obs, times):
        """
        Args:
            times (int): how many times to check
        Returns:
            (int, int): (x, y)
        """

        for i in range(times):
            build_xy = (MAIN_COMMAND_CENTER_POTISION[self.base_top_left]['x'] + random.randint(BUILD_RANGE_MIN, BUILD_RANGE_MAX) * random.choice([-1, 1]),
                        MAIN_COMMAND_CENTER_POTISION[self.base_top_left]['y'] + random.randint(BUILD_RANGE_MIN, BUILD_RANGE_MAX) * random.choice([-1, 1]))                
            if self.check_if_buildable(obs, *build_xy):
                break
        return build_xy
                 
#build something
    def build_unit(self, obs, building):
        if building.find("TechLab") == -1 and building.find("Reactor") == -1:
            scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
            if len(scvs) <= 0:
                return actions.RAW_FUNCTIONS.no_op()
            
            if building == "CommandCenter":
                command_centers = self.get_my_units_by_type(obs, units.Terran.CommandCenter)
                #(19, 23)(41, 21)(17, 48)(39, 45)
                if len(command_centers) == 0 and len(scvs) > 0:
                    build_xy = (19, 23) if self.base_top_left else (39, 45)
                elif len(command_centers) == 1 and len(scvs) > 0:
                    build_xy = (41, 21) if self.base_top_left else (17, 48)
                elif len(command_centers) == 2 and len(scvs) > 0:
                    build_xy = (17, 48) if self.base_top_left else (41, 21)
                else:
                    return actions.RAW_FUNCTIONS.no_op()
            elif building == "Refinery":
                gas_patches = [unit for unit in obs.observation.raw_units
                            if unit.unit_type == units.Neutral.VespeneGeyser ]
                scv = random.choice(scvs)
                distances = self.get_distances(obs, gas_patches, (scv.x, scv.y))
                gas_patch = gas_patches[np.argmin(distances)]
                return actions.RAW_FUNCTIONS.Build_Refinery_pt("now", scv.tag, gas_patch.tag)
            else: 
                ### buildings which only research
                if building in ["GhostAcademy", "EngineeringBay", "Armory", "FusionCore"]:
                    research_building_list = self.get_my_units_by_type(obs, getattr(units.Terran, building))
                    if len(research_building_list) > 0:
                        return actions.RAW_FUNCTIONS.no_op()

                build_xy = self.get_proper_position_to_build(obs, 10)

            distances = self.get_distances(obs, scvs, build_xy)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.__getattr__(f"Build_{building}_pt")(
                    "now", scv.tag, build_xy)

        else:
            # TechLab or Reactor
            TorR = "TechLab" if building.find("TechLab") != -1 else "Reactor"
            base_building_name = building[building.find("_")+1:building.find(TorR)]
            base_building_list = self.get_my_completed_units_by_type(
                obs, getattr(units.Terran, base_building_name))
            if len(base_building_list) > 0:
                return actions.RAW_FUNCTIONS.__getattr__(f"Build_{TorR}_{base_building_name}_quick")("now", [b.tag for b in base_building_list])

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
        self.previous_building_num = 0
        self.saved_reward = 0

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

    def get_state(self, obs):

        player_mineral = obs.observation.player.minerals
        player_vespene = obs.observation.player.vespene

        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        idle_scvs = [scv for scv in scvs if scv.order_length == 0]
        buildings = [len(self.get_my_completed_units_by_type(obs, getattr(units.Terran, unit))) for unit in MY_BUILDING_LIST] 


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
                step_reward = self.get_reward(obs, action)

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

    def get_prev_reward(self, obs):
        reward = 0
        total_value_units_score = obs.observation.score_cumulative.total_value_units
        total_value_structures_score = obs.observation.score_cumulative.total_value_structures
        total_spent_minerals = obs.observation.score_cumulative.spent_minerals
        total_spent_vespene = obs.observation.score_cumulative.spent_vespene
        player_mineral = obs.observation.player.minerals
        player_vespene = obs.observation.player.vespene
        building_num = len(self.get_my_building_by_pos(obs))

        # If spent mineral from prev to now state, get positive reward
        if total_spent_minerals > self.previous_total_spent_minerals:
            reward += MORE_MINERALS_USED_REWARD_RATE * \
                (total_spent_minerals - self.previous_total_spent_minerals)
        # If spent vespene from prev to now state, get positive reward
        if total_spent_vespene > self.previous_total_spent_vespene:
            reward += MORE_VESPENE_USED_REWARD_RATE * \
                (total_spent_vespene - self.previous_total_spent_vespene)
        
        # If too much mineral in this state, get negative reward
        if player_mineral > 1000:
            reward -= TOO_MUCH_MINERAL_PENALTY * (player_mineral - 1000)
        # If too much mineral in this state, get negative reward
        if player_vespene > 500:
            reward -= TOO_MUCH_VESPENE_PENALTY * (player_vespene - 500)
        
        # If lose building in this step, get negative reward
        if building_num < self.previous_building_num:
            reward += FAILED_BUILDING * (building_num - self.previous_building_num)
          
        self.previous_total_value_units_score = total_value_units_score
        self.previous_total_value_structures_score = total_value_structures_score
        self.previous_total_spent_minerals = total_spent_minerals
        self.previous_total_spent_vespene = total_spent_vespene
        self.previous_building_num = building_num
        return reward

    def get_saved_reward(self, obs, action):
        reward = 0
        player_mineral = obs.observation.player.minerals
        player_vespene = obs.observation.player.vespene
        free_supply = (obs.observation.player.food_cap - obs.observation.player.food_used)

        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        idle_scvs = [scv for scv in scvs if scv.order_length == 0]

        # If this state choose a action that can't execute, get negative reward
        if self.can_afford_unit(obs, action) == False:
            reward += FAILED_COMMAND

        if action == "harvest_minerals":
            if len(idle_scvs) == 0:
                reward += FAILED_COMMAND
        elif action == "train_SCV":
            completed_commandcenters = self.get_my_completed_units_by_type(obs, units.Terran.CommandCenter)
            if len(completed_commandcenters) <= 0 or free_supply == 0 or player_mineral < 50:
                reward += FAILED_COMMAND
        elif action == "harvest_gas":
            notfull_completed_refinerys = self.get_notfull_worker_building_by_type(obs, units.Terran.Refinery)
            if len(scvs) <= 0 or len(notfull_completed_refinerys) <= 0:
                reward += FAILED_COMMAND

        return reward

    def get_reward(self, obs, action):
        prev_reward = self.get_prev_reward(obs)
        step_reward = prev_reward - self.saved_reward
        self.saved_reward = self.get_saved_reward(obs, action)

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
