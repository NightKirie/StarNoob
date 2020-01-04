from base_agent import *

import unit.terran_unit as terran

DATA_FILE = 'Sub_training_data'

FAILED_COMMAND = 0.01
TRAIN_UNIT_REWARD_RATE = 0.01
MORE_MINERALS_USED_REWARD_RATE = 0.001
MORE_VESPENE_USED_REWARD_RATE = 0.002
TOO_MUCH_MINERAL_PENALTY = 0.005
TOO_MUCH_VESPENE_PENALTY = 0.01

SAVE_POLICY_NET = 'model/training_dqn_policy'
SAVE_TARGET_NET = 'model/training_dqn_target'
SAVE_MEMORY = 'model/training_memory'

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 500

class Agent(BaseAgent):

    actions = tuple(["do_nothing"]) + \
        tuple([f"train_{unit}" for unit in MY_ARMY_LIST])

    def __init__(self):
        super(Agent, self).__init__()

        # Create action function
        for unit in MY_ARMY_LIST:
            self.__setattr__(
                f"train_{unit}", partial(
                    self.train_unit, unit=unit))

    def get_distances(self, obs, units, xy):
        units_xy = [(unit.x, unit.y) for unit in units]
        return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1)


    def get_least_busy_building(self, obs, unit_type):
        """ get least busy building
        Args:
          obs
          unit_type (int): from pysc2.lib.units.Terran.XXX
        Returns:
          building.tag (int): if building exist
          or
          False: if building not exist
        """
        completed_buildinges = self.get_my_completed_units_by_type(
            obs, unit_type)
        if len(completed_buildinges) > 0:
            return min(completed_buildinges, key=lambda building: building.order_length)
        else:
            return False

    def can_afford_unit(self, obs, unit):
        """ Check if all condition can afford to this unit
        Args:
          obs
          unit_type (object): from unit.terran_unit.XXX
        Returns:
          True: if can afford
          or
          False: if can not afford
        """
        can_afford = False
        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)
        if isinstance(unit, terran.TerranCreature):
            if free_supply > 0 and \
               obs.observation.player.minerals >= unit.mineral_price and \
               obs.observation.player.vespene >= unit.vespene_price and \
               0 not in [len(self.get_my_completed_units_by_type(obs, getattr(terran, build_from)().index)) for build_from in unit.build_from] and \
               0 not in [len(self.get_my_completed_units_by_type(obs, getattr(terran, requirement)().index)) for requirement in unit.requirements]:
                can_afford = True
        return can_afford

    def train_unit(self, obs, unit=None):
        """ Try to train specific unit
        Args:
          obs
          unit (string): can pass to unit.terran_unit.XXX to get unit information
        Returns:
          train unit action: if can train
          or
          no-op: if can not train
        """
        unit = getattr(terran, unit)()

        building_tag = False
        requirement_flag = False

        # Get if all condition match to train the target unit
        if self.can_afford_unit(obs, unit):
            building_tag =  [self.get_least_busy_building(obs, getattr(terran, build_from)().index) for build_from in unit.build_from][0].tag
            return actions.RAW_FUNCTIONS.__getattr__(
                f"Train_{unit.name}_quick")("now", building_tag)
        else:
            return actions.RAW_FUNCTIONS.no_op()


class SubAgent_Training(Agent):

    def __init__(self):
        super(SubAgent_Training, self).__init__()
        self.new_game()
        self.set_DQN(SAVE_POLICY_NET, SAVE_TARGET_NET, SAVE_MEMORY)
        self.episode = 0


    def reset(self):
        log.debug('in reset')
        super(SubAgent_Training, self).reset()
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
        self.saved_reward = 0

    def get_state(self, obs):
        complete_trainable_building = [len(self.get_my_completed_units_by_type(obs, getattr(terran, building)().index)) for building in TRAINABLE_BUILDING]
        complete_unit = [len(self.get_my_units_by_type(obs, getattr(terran, unit)().index)) for unit in MY_ARMY_LIST]
        completed_supply_depots = self.get_my_completed_units_by_type(
            obs, terran.SupplyDepot().index)

        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)

        return tuple([self.base_top_left,
                    len(completed_supply_depots),
                    free_supply] +
                    complete_trainable_building +
                    complete_unit)




    def step(self, obs):
        super(SubAgent_Training, self).step(obs)

        self.episode += 1
        state = self.get_state(obs)
        log.debug(state)
        action, action_idx = self.select_action(state)
        log.info(action)

        if self.episodes % 3 == 0:
            if self.previous_action is not None:
                step_reward = self.get_reward(obs, action)
                log.log(LOG_REWARD, "training reward = " + str(step_reward))
                if not obs.last():
                    self.memory.push(torch.Tensor(self.previous_state).to(device),
                                    torch.LongTensor([self.previous_action_idx]).to(device),
                                    torch.Tensor(state).to(device),
                                    torch.Tensor([step_reward]).to(device))

                    self.optimize_model()
                else:
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

        prev_reward = 0

        # If train valueable unit, get positive reward
        if total_value_units_score > self.previous_total_value_units_score:
            prev_reward += TRAIN_UNIT_REWARD_RATE * \
                (total_value_units_score - self.previous_total_value_units_score)

        # If spent mineral from prev to now state, get positive reward
        if total_spent_minerals > self.previous_total_spent_minerals:
            prev_reward += MORE_MINERALS_USED_REWARD_RATE * \
                (total_spent_minerals - self.previous_total_spent_minerals)
        # If spent vespene from prev to now state, get positive reward
        if total_spent_vespene > self.previous_total_spent_vespene:
            prev_reward += MORE_VESPENE_USED_REWARD_RATE * \
                (total_spent_vespene - self.previous_total_spent_vespene)
        
        # If too much mineral in this state, get negative reward
        if player_mineral > 1000:
            reward -= TOO_MUCH_MINERAL_PENALTY * (player_mineral - 1000)
        # If too much mineral in this state, get negative reward
        if player_vespene > 500:
            reward -= TOO_MUCH_VESPENE_PENALTY * (player_vespene - 500)

        self.previous_total_value_units_score = total_value_units_score
        self.previous_total_value_structures_score = total_value_structures_score
        self.previous_total_spent_minerals = total_spent_minerals
        self.previous_total_spent_vespene = total_spent_vespene
        return reward

    def get_saved_reward(self, obs, action):
        reward = 0
        if action.find("train") != -1 :
            unit_name = action[action.index("_")+1:]
            if not self.can_afford_unit(obs, getattr(terran, unit_name)()):
                reward = FAILED_COMMAND            
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
            log.log(LOG_MODEL, "Save memory in training")
