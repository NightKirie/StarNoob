import threading
import time
import importlib
import sys
import os

STARNOOB_LIB_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '../../../lib')

sys.path.append(STARNOOB_LIB_DIR)


from pysc2 import maps
from pysc2.env import available_actions_printer
from pysc2.env import run_loop
from pysc2.env import sc2_env
from pysc2.lib import point_flag
from pysc2.lib import stopwatch

from absl import app
from absl import flags

from rl.deep_q_learning import BaseRLAgent as Agent
import env_config as config

def run_thread(players):
    with sc2_env.SC2Env(
        map_name=config.FLAGS.map_name,
        battle_net_map=config.FLAGS.battle_net_map,
        players=players,
        agent_interface_format=sc2_env.parse_agent_interface_format(
            feature_screen=config.FLAGS.feature_screen_size,
            feature_minimap=config.FLAGS.feature_minimap_size,
            rgb_screen=config.FLAGS.rgb_screen_size,
            rgb_minimap=config.FLAGS.rgb_minimap_size,
            action_space=config.FLAGS.action_space,
            use_feature_units=config.FLAGS.use_feature_units
        ),
        discount=config.FLAGS.discount,
        discount_zero_after_timeout=config.FLAGS.discount_zero_after_timeout,
        visualize=config.FLAGS.visualize,
        step_mul=config.FLAGS.step_mul,
        realtime=config.FLAGS.realtime,
        save_replay_episodes=config.FLAGS.save_replay_episodes,
        replay_dir=config.FLAGS.replay_dir,
        replay_prefix=config.FLAGS.replay_prefix,
        game_steps_per_episode=config.FLAGS.game_steps_per_episode,
        score_index=config.FLAGS.score_index,
        score_multiplier=config.FLAGS.score_multiplier,
        random_seed=config.FLAGS.random_seed,
        disable_fog=config.FLAGS.disable_fog,
        ensure_available_actions=config.FLAGS.ensure_available_actions) as env:

        env = available_actions_printer.AvailableActionsPrinter(env)
        agent = Agent()
        # run_loop([agent], env, FLAGS.max_agent_steps)
        agent.train(env, True)


def main(unused_argv):
    """Run an agent."""
    players = [sc2_env.Agent(sc2_env.Race.terran),
                sc2_env.Bot(sc2_env.Race.terran,
                sc2_env.Difficulty.very_easy)]

    if config.FLAGS.profile or config.FLAGS.trace:
        stopwatch.sw.enabled()
    else:
        stopwatch.sw.disable()

    if config.FLAGS.trace:
        stopwatch.sw.trace()

    maps.get(config.FLAGS.map_name)  # Assert the map exists.
    run_thread(players)

    if config.FLAGS.profile:
        print(stopwatch.sw)


def entry_point():  # Needed so setup.py scripts work.
    app.run(main)


if __name__ == "__main__":
    app.run(main)
