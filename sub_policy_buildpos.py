from base_agent import *

DATA_FILE = 'Sub_buildpos_data'

LOST_STRUCTURE_RATE = 0.0004
GET_STRUCTURE_RATE = 0.0004

SUB_BUILD_DIVISION = 64
SUB_BUILDPOS_SIZE = 1

SUB_LOCATION_DIVISION = 16
SUB_LOCATION_SIZE = 4

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 500

SAVE_POLICY_NET = 'model/buildpos_dqn_policy'
SAVE_TARGET_NET = 'model/buildpos_dqn_target'
SAVE_MEMORY = 'model/buildpos_memory'

class Agent(BaseAgent):
    actions = tuple(["do_nothing"]) + \
              tuple([f"buildpos_{i}_{j}" for i in range(0, SUB_BUILD_DIVISION) for j in range(0, SUB_BUILD_DIVISION)])


    def __init__(self):
        super().__init__()

        for i in range(0, SUB_BUILD_DIVISION):
            for j in range(0, SUB_BUILD_DIVISION):
                self.__setattr__(
                    f"buildpos_{i}_{j}", partial(
                        self.buildpos, range=SimpleNamespace(**{'x': i, 'y': j}), size=SUB_BUILD_SIZE))

    def buildpos(self, obs, range, size):

        buildpos = SimpleNamespace(**{'x': range.x * SUB_BUILDPOS_SIZE, 'y': range.y * SUB_BUILDPOS_SIZE})
        return (buildpos.x, buildpos.y)



class SubAgent_Buildpos(Agent):

    def __init__(self):
        super(SubAgent_Buildpos, self).__init__()
        self.new_game()
        self.set_DQN(SAVE_POLICY_NET, SAVE_TARGET_NET, SAVE_MEMORY)
        self.episode = 0

    def reset(self):
        super(SubAgent_Buildpos, self).reset()
        self.new_game()

    def new_game(self):
        self.base_top_left = None
        self.previous_state = None
        self.previous_action = None
        self.previous_my_structures = 0
        self.now_reward = 0

    def get_state(self, obs=None):

        my_building_location = [self.get_my_building_by_pos(obs,
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


        return tuple([self.base_top_left]) + \
                    tuple([len(my_building_location[i * SUB_LOCATION_DIVISION + j]) for i in range(0, SUB_LOCATION_DIVISION) for j in range(0, SUB_LOCATION_DIVISION)]) + \
                    tuple([len(enemy_building_location[i * SUB_LOCATION_DIVISION + j]) for i in range(0, SUB_LOCATION_DIVISION) for j in range(0, SUB_LOCATION_DIVISION)])

    def step(self, obs):

        super(SubAgent_Buildpos, self).step(obs)

        self.episode += 1
        state = self.get_state(obs)
        log.debug(f"state: {state}")
        action, action_idx = self.select_action(state)
        log.info(action)


        if self.episodes % 3 == 1:
            if self.previous_action is not None:
                step_reward = self.get_reward(obs)
                log.log(LOG_REWARD, "buildpos reward = " + str(obs.reward +step_reward))
                if not obs.last():
                    self.memory.push(torch.Tensor(self.previous_state).to(device),
                                    torch.LongTensor([self.previous_action_idx]).to(device),
                                    torch.Tensor(state).to(device),
                                    torch.Tensor([obs.reward + step_reward]).to(device))

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

        x = 0
        y = 0
        if '_' in action:
            cut_action, x, y = action.split('_')
        posdict = {'x':x, 'y':y}
        return posdict

    def get_reward(self, obs):
        my_structures = len(self.get_my_building_by_pos(obs))

        prev_reward = 0

        # If get unit in buildpos, get negative reward
        if my_structures - self.previous_my_structures > 0:
            prev_reward += LOST_STRUCTURE_RATE * (self.previous_my_structures - my_structures)

        # If loss stuctures in buildpos, get negative reward
        if my_structures - self.previous_my_structures < 0:
            prev_reward -= LOST_STRUCTURE_RATE * (self.previous_my_structures - my_structures)

        ## Update reward
        step_reward = prev_reward - self.now_reward

        ## Now reward will update in next epoch

        self.previous_my_structures = my_structures
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
            log.log(LOG_MODEL, "Save memory in buildpos")
