from base_agent import *

DATA_FILE = 'Sub_battle_data'
KILL_UNIT_REWARD_RATE = 0.02
KILL_BUILDING_REWARD_RATE = 0.04
DEAD_UNIT_REWARD_RATE = 0.001 * 0
DEAD_BUILDING_REWARD_RATE = 0.002 * 0
DEALT_TAKEN_REWARD_RATE = 0.001
FOUND_ENEMY_RATE = 0.001
LOST_UNIT_RATE = 0.02
LOST_STRUCTURE_RATE = 0.04

SUB_ATTACK_DIVISION = 4
SUB_ATTACK_SIZE = 16

SUB_LOCATION_DIVISION = 4
SUB_LOCATION_SIZE = 16

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 500

SAVE_POLICY_NET = 'model/battle_dqn_policy'
SAVE_TARGET_NET = 'model/battle_dqn_target'
SAVE_MEMORY = 'model/battle_memory'

Point = namedtuple("Vector_tuple", ["x", "y"])


class Agent(BaseAgent):
    actions = tuple(["do_nothing"]) + \
              tuple([f"attack_point_{i}_{j}" for i in range(0, SUB_ATTACK_DIVISION) for j in range(0, SUB_ATTACK_DIVISION)]) + \
              tuple(["attack_enemy"]) 

    def __init__(self):
        super().__init__()

        for i in range(0, SUB_ATTACK_DIVISION):
            for j in range(0, SUB_ATTACK_DIVISION):
                self.__setattr__(
                    f"attack_point_{i}_{j}", partial(
                        self.attack_point, point=Point(i, j), size=SUB_ATTACK_SIZE))

    def attack_point(self, obs, point, size):
        armys = self.get_my_army_by_pos(obs)
        if len(armys) > 0:
            if self.base_top_left == 0:
                attack = (point.x * SUB_ATTACK_SIZE + SUB_LOCATION_SIZE / 2, point.y * SUB_ATTACK_SIZE + SUB_LOCATION_SIZE / 2)
            else: 
                attack = (64 - (point.x * SUB_ATTACK_SIZE + SUB_LOCATION_SIZE / 2), 64 - (point.y * SUB_ATTACK_SIZE + SUB_LOCATION_SIZE / 2))
            return actions.RAW_FUNCTIONS.Attack_pt(
                "now", [soldier.tag for soldier in armys], attack)
        return actions.RAW_FUNCTIONS.no_op()

    def attack_enemy(self, obs): 
        """ attack enemy in pos """
        enemy_army_list = self.get_enemy_army_by_pos(obs)
        enemy_building_list = self.get_enemy_building_by_pos(obs)
        armys = self.get_my_army_by_pos(obs)
        if len(armys) > 0 and (enemy_army_list != [] or enemy_building_list != []):
            if enemy_army_list != []:
                distances = self.get_distances(obs, enemy_army_list, (armys[0].x, armys[0].y))
                attack_enemy = enemy_army_list[np.argmin(distances)]
            elif enemy_building_list != []:
                distances = self.get_distances(obs, enemy_building_list, (armys[0].x, armys[0].y))
                attack_enemy = enemy_building_list[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Attack_unit(
                "now", [soldier.tag for soldier in armys], attack_enemy.tag)
            # if enemy_army_list != []:
            #     attack = Point(enemy_army_list[0].x, enemy_army_list[0].y)
            # elif enemy_building_list != []:
            #     attack = Point(enemy_building_list[0].x, enemy_building_list[0].y)
            # return actions.RAW_FUNCTIONS.Attack_pt(
            #     "now", [soldier.tag for soldier in armys], attack)
        return actions.RAW_FUNCTIONS.no_op()


class SubAgent_Battle(Agent):

    def __init__(self):
        super(SubAgent_Battle, self).__init__()
        self.new_game()
        self.set_DQN(SAVE_POLICY_NET, SAVE_TARGET_NET, SAVE_MEMORY)
        self.episode = 0

    def reset(self):
        super(SubAgent_Battle, self).reset()
        self.new_game()

    def new_game(self):
        self.base_top_left = None
        self.previous_state = None
        self.previous_action = None
        self.previous_total_value_units_score = 0
        self.previous_total_value_structures_score = 0
        self.previous_total_killed_value_units_score = 0
        self.previous_total_killed_value_structures_score = 0
        self.previous_total_damage_dealt = 0
        self.previous_total_damage_taken = 0
        self.previous_my_units = 0
        self.previous_my_structures = 0
        self.saved_reward = 0

    def get_state(self, obs=None):

        my_army_location = [self.get_my_army_by_pos(obs,
                                                    i * SUB_LOCATION_SIZE,
                                                    j * SUB_LOCATION_SIZE,
                                                    (i + 1) * SUB_LOCATION_SIZE,
                                                    (j + 1) * SUB_LOCATION_SIZE)
                                                    for i in range(0, SUB_LOCATION_DIVISION) for j in range(0, SUB_LOCATION_DIVISION)]
        my_building_location = [self.get_my_building_by_pos(obs,
                                                    i * SUB_LOCATION_SIZE,
                                                    j * SUB_LOCATION_SIZE,
                                                    (i + 1) * SUB_LOCATION_SIZE,
                                                    (j + 1) * SUB_LOCATION_SIZE)
                                                    for i in range(0, SUB_LOCATION_DIVISION) for j in range(0, SUB_LOCATION_DIVISION)]

        enemy_army_location = [self.get_enemy_army_by_pos(obs,
                                                    i * SUB_LOCATION_SIZE,
                                                    j * SUB_LOCATION_SIZE,
                                                    (i + 1) * SUB_LOCATION_SIZE,
                                                    (j + 1) * SUB_LOCATION_SIZE)
                                                    for i in range(0, SUB_LOCATION_DIVISION) for j in range(0, SUB_LOCATION_DIVISION)]
        enemy_building_location = [self.get_enemy_building_by_pos(obs,
                                                    i * SUB_LOCATION_SIZE,
                                                    j * SUB_LOCATION_SIZE,
                                                    (i + 1) * SUB_LOCATION_SIZE,
                                                    (j + 1) * SUB_LOCATION_SIZE)
                                                    for i in range(0, SUB_LOCATION_DIVISION) for j in range(0, SUB_LOCATION_DIVISION)]

        free_supply = (obs.observation.player.food_cap -
                    obs.observation.player.food_used)

        player_food_army = obs.observation.player.food_army

        return tuple([self.base_top_left,
                    free_supply,
                    player_food_army]) + \
                    tuple([len(my_army_location[i * SUB_LOCATION_DIVISION + j]) for i in range(0, SUB_LOCATION_DIVISION) for j in range(0, SUB_LOCATION_DIVISION)]) + \
                    tuple([len(my_building_location[i * SUB_LOCATION_DIVISION + j]) for i in range(0, SUB_LOCATION_DIVISION) for j in range(0, SUB_LOCATION_DIVISION)]) + \
                    tuple([len(enemy_army_location[i * SUB_LOCATION_DIVISION + j]) for i in range(0, SUB_LOCATION_DIVISION) for j in range(0, SUB_LOCATION_DIVISION)]) + \
                    tuple([len(enemy_building_location[i * SUB_LOCATION_DIVISION + j]) for i in range(0, SUB_LOCATION_DIVISION) for j in range(0, SUB_LOCATION_DIVISION)])

    def step(self, obs):

        super(SubAgent_Battle, self).step(obs)

        self.episode += 1
        state = self.get_state(obs)
        log.debug(f"state: {state}")
        action, action_idx = self.select_action(state)
        log.info(action)

        if self.episodes % 3 == 1:
            if self.previous_action is not None:
                step_reward = self.get_reward(obs, action)
                log.log(LOG_REWARD, "battle reward = " + str(step_reward))
                if not obs.last():
                    self.memory.push(torch.Tensor(self.previous_state).to(device),
                                    torch.LongTensor([self.previous_action_idx]).to(device),
                                    torch.Tensor(state).to(device),
                                    torch.Tensor([step_reward]).to(device))

                    self.optimize_model()
                else:
                    pass
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
        total_killed_value_units_score = obs.observation.score_cumulative.killed_value_units
        total_killed_value_structures_score = obs.observation.score_cumulative.killed_value_structures
        total_damage_dealt = obs.observation.score_by_vital.total_damage_dealt[0]
        total_damage_taken = obs.observation.score_by_vital.total_damage_taken[0]
        visiable_enemy_army = len(self.get_enemy_army_by_pos(obs))
        visiable_enemy_structures = len(self.get_enemy_building_by_pos(obs))
        my_army = len(self.get_my_army_by_pos(obs))
        my_structures = len(self.get_my_building_by_pos(obs))

        # If kill a more valuable unit, get positive reward
        if total_killed_value_units_score > self.previous_total_killed_value_units_score:
            reward += KILL_UNIT_REWARD_RATE * \
                (total_killed_value_units_score -
                    self.previous_total_killed_value_units_score)

        # If kill a more valuable structure, get positive reward
        if total_killed_value_structures_score > self.previous_total_killed_value_structures_score:
            reward += KILL_BUILDING_REWARD_RATE * \
                (total_killed_value_structures_score -
                    self.previous_total_killed_value_structures_score)

        # If in this epoch, dealt damage is more than taken damage, get positive reward
        if total_damage_dealt - self.previous_total_damage_dealt > total_damage_taken - self.previous_total_damage_taken:
            reward += DEALT_TAKEN_REWARD_RATE * \
                ((total_damage_dealt - self.previous_total_damage_dealt) -
                    (total_damage_taken - self.previous_total_damage_taken))
        # If in this epoch, dealt damage is less than taken damage, get negative reward
        else:
            reward -= DEALT_TAKEN_REWARD_RATE * \
                ((total_damage_taken - self.previous_total_damage_taken) - 
                (total_damage_dealt - self.previous_total_damage_dealt))

        # If in this epoch, found enemy, get positive reward
        if visiable_enemy_army > 0 or visiable_enemy_structures > 0:
            reward += FOUND_ENEMY_RATE * \
                (visiable_enemy_army + visiable_enemy_structures)

        # If loss unit in battle, get negative reward
        if my_army - self.previous_my_units < 0:
            reward -= LOST_UNIT_RATE * (self.previous_my_units - my_army)

        # If loss stuctures in battle, get negative reward
        if my_structures - self.previous_my_structures < 0:
            reward -= LOST_STRUCTURE_RATE * (self.previous_my_structures - my_structures)
        
        self.previous_total_value_units_score = total_value_units_score
        self.previous_total_value_structures_score = total_value_structures_score
        self.previous_killed_unit_score = total_killed_value_units_score
        self.previous_total_killed_value_structures_score = total_killed_value_structures_score
        self.previous_total_damage_dealt = total_damage_dealt
        self.previous_total_damage_taken = total_damage_taken
        self.previous_my_units = my_army
        self.previous_my_structures = my_structures
        return reward

    def get_saved_reward(self, obs, action):
        reward = 0

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
            log.log(LOG_MODEL, "Save memory in battle")