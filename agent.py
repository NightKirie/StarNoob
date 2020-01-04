from pysc2.env import sc2_env, run_loop

from base_agent import *
import sub_policy_battle
import sub_policy_economic
import sub_policy_training

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 500

SAVE_POLICY_NET = 'model/agent_dqn_policy'
SAVE_TARGET_NET = 'model/agent_dqn_target'
SAVE_MEMORY = 'model/agent_memory'
TIME_PENALTY_Q = 0.0001

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

        



# class RandomAgent(Agent):
#     def step(self, obs):
#         super(RandomAgent, self).step(obs)
#         action = random.choice(self.actions)
#         return getattr(self, action)(obs)


class SmartAgent(Agent):

    def __init__(self):
        log.debug('in __init__')
        super(SmartAgent, self).__init__()
        self.new_game()
        self.set_DQN(SAVE_POLICY_NET, SAVE_TARGET_NET, SAVE_MEMORY)
        self.episode = 0

    def reset(self):
        log.debug('in reset')
        if self.episodes != 0:
            log.log(LOG_EPISODE,
                    f"Episode {self.episodes} finished after {self.steps} game steps. Score: {self.score}. Reward: {self.reward}")
        super(SmartAgent, self).reset()
        log.log(LOG_EPISODE, f"Starting episode {self.episodes}")
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
        self.time_penalty = 0
        self.steps = 0
        self.score = 0

    def get_state(self, obs):
        """
        get state of starcraft II
        Args:  observation
        Returns: state(list)
        """

        player_mineral = obs.observation.player.minerals
        player_vespene = obs.observation.player.vespene

        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        idle_scvs = [scv for scv in scvs if scv.order_length == 0]

        player_food_used = obs.observation.player.food_used
        player_food_cap = obs.observation.player.food_cap
        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)
        player_food_army = obs.observation.player.food_army
        player_food_workers = obs.observation.player.food_workers

        my_army_num = self.get_my_army_by_pos(obs)
        my_building_num = self.get_my_building_by_pos(obs)
        enemy_army_num = self.get_enemy_army_by_pos(obs)
        enemy_building_num = self.get_enemy_building_by_pos(obs)

        return (self.base_top_left,
                player_mineral,
                player_vespene,
                player_food_used,
                player_food_cap,
                free_supply,
                player_food_army,
                player_food_workers,
                len(scvs),
                len(idle_scvs),
                len(my_army_num),
                len(my_building_num),
                len(enemy_army_num),
                len(enemy_building_num)
                )

    def step(self, obs):

        """
        every step starcraft II will call this function
        return: getattr(self, action)(obs)
        """
        log.debug('into step')
        super(SmartAgent, self).step(obs)
        if obs.first():
            command_center = self.get_my_units_by_type(
                obs, units.Terran.CommandCenter)[0]
            self.base_top_left = (command_center.x < 32)
            self.battle_policy.set_top_left(obs)
            self.economic_policy.set_top_left(obs)
            self.training_policy.set_top_left(obs)

        self.episode += 1
        state = self.get_state(obs)
        log.debug(f"state: {state}")
        action, action_idx = self.select_action(state)
        log.info(action)

        if self.previous_action is not None:
            step_reward = self.get_reward(obs)
            log.log(LOG_REWARD, "agent reward = " + str(obs.reward + step_reward))
            if not obs.last():
                self.memory.push(torch.Tensor(self.previous_state).to(device),
                                 torch.LongTensor([self.previous_action_idx]).to(device),
                                 torch.Tensor(state).to(device),
                                 torch.Tensor([obs.reward + step_reward]).to(device))
                self.optimize_model()
            else:
                # save models
                if SAVE_MODEL:
                    self.save_module()
                    self.training_policy.save_module()
                    self.economic_policy.save_module()
                    self.battle_policy.save_module()
                return
        else:
            pass

        self.previous_state = state
        self.previous_action = action
        self.previous_action_idx = action_idx

        # record score for episode ending use
        self.score = obs.observation.score_cumulative.score
        log.debug('get out step')
        return getattr(self, action)(obs)

    def get_reward(self, obs):
        log.info(self.battle_policy.get_reward(obs, self.battle_policy.previous_action))
        reward = 0
        reward += -TIME_PENALTY_Q
        return reward

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
            log.log(LOG_MODEL, "Save memory in agent")


def main(unused_argv):
    agent1 = SmartAgent()
    #agent2 = RandomAgent()
    try:
        with sc2_env.SC2Env(
            map_name=FLAGS.map_name,
            battle_net_map=FLAGS.battle_net_map,
            players=FLAGS.players,
            agent_interface_format=sc2_env.parse_agent_interface_format(
                feature_screen=FLAGS.feature_screen_size,
                feature_minimap=FLAGS.feature_minimap_size,
                rgb_screen=FLAGS.rgb_screen_size,
                rgb_minimap=FLAGS.rgb_minimap_size,
                use_raw_units=FLAGS.use_raw_units,
                use_raw_actions=FLAGS.use_raw_actions,
                raw_resolution=FLAGS.raw_resolution,
            ),
            discount=FLAGS.discount,
            discount_zero_after_timeout=FLAGS.discount_zero_after_timeout,
            visualize=FLAGS.visualize,
            step_mul=FLAGS.step_mul,
            realtime=FLAGS.realtime,
            save_replay_episodes=FLAGS.save_replay_episodes,
            replay_dir=FLAGS.replay_dir,
            replay_prefix=FLAGS.replay_prefix,
            game_steps_per_episode=FLAGS.game_steps_per_episode,
            score_index=FLAGS.score_index,
            score_multiplier=FLAGS.score_multiplier,
            random_seed=FLAGS.random_seed,
            disable_fog=FLAGS.disable_fog,
            ensure_available_actions=FLAGS.ensure_available_actions
        ) as env:
            #run_loop.run_loop([agent1, agent2], env, max_episodes=1000)
            run_loop.run_loop([agent1], env, max_episodes=1000)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    app.run(main)
