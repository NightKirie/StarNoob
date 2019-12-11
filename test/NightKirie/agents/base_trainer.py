import threading
import time
import importlib


from pysc2 import maps
from pysc2.env import available_actions_printer
from pysc2.env import run_loop
from pysc2.env import sc2_env
from pysc2.lib import point_flag
from pysc2.lib import stopwatch

from absl import app
from absl import flags

from lib.rl.deep_q_learning import BaseRLAgent as Agent
import env_config as config

def run_thread(players, visualize):
    with sc2_env.SC2Env(
        map_name=config.FLAGS.map_name,
        battle_net_map=config.FLAGS.battle_net_map,
        players=None,
        agent_interface_format=None,
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
        ensure_available_actions=config.FLAGS.ensure_available_actions,
        version=config.FLAGS.version) as env:

        env = available_actions_printer.AvailableActionsPrinter(env)
        agent = Agent()
        # run_loop([agent], env, FLAGS.max_agent_steps)
        run_loop.run_loop(agents, env, config.FLAGS.max_agent_steps, config.FLAGS.max_episodes)


def main(unused_argv):
    """Run an agent."""
    if config.FLAGS.profile or config.FLAGS.trace:
        stopwatch.sw.enabled()
    else:
        stopwatch.sw.disable()

    if config.FLAGS.trace:
        stopwatch.sw.trace()

    maps.get(config.FLAGS.map_name)  # Assert the map exists.
    run_thread(config.FLAGS.map, config.FLAGS.render)

    if config.FLAGS.profile:
        print(stopwatch.sw)


def entry_point():  # Needed so setup.py scripts work.
    app.run(main)


if __name__ == "__main__":
    app.run(main)
