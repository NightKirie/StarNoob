import threading
import time

from pysc2 import maps
from pysc2.env import available_actions_printer
# from pysc2.env import run_loop
from pysc2.env import sc2_env
from pysc2.lib import stopwatch

from absl import app
from absl import flags

from StarNoob.lib.rl.deep_q_learning import BaseRLAgent as Agent

FLAGS = flags.FLAGS
flags.DEFINE_string("map_name", "Simple64", "Name of a SC2 map")


flags.DEFINE_bool("render", False, "Whether to render with pygame.")
flags.DEFINE_bool("train", False, "Whether we are training or running")
flags.DEFINE_integer("screen_resolution", 28,
                     "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 64,
                     "Resolution for minimap feature layers.")

# flags.DEFINE_integer("max_agent_steps", 2500, "Total agent steps.")
flags.DEFINE_integer("game_steps_per_episode", 0, "Game steps per episode.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")

# flags.DEFINE_string("agent", "pysc2.agents.random_agent.RandomAgent", "Which agent to run")
flags.DEFINE_enum("agent_race", None, sc2_env.Race._member_names_, "Agent's race.")
flags.DEFINE_enum("bot_race", None, sc2_env.Race._member_names_, "Bot's race.")
flags.DEFINE_enum("difficulty", None, sc2_env.Difficulty._member_names_,
                  "Bot's strength.")

flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")

flags.DEFINE_bool("save_replay", False, "Whether to save a replay at the end.")



def run_thread(players, visualize):
    with sc2_env.SC2Env(
        map_name=FLAGS.map_name,
        players=None,
        agent_interface_format=None,
        visualize=False,
        step_mul=None,
        realtime=False,
        save_replay_episodes=0,
        replay_dir=None,
        replay_prefix=None,
        game_steps_per_episode=None,
        score_index=None,
        score_multiplier=None,
        random_seed=None,
        disable_fog=False,
        ensure_available_actions=True,
        version=None

        """ map_name=map_name,
        agent_race=FLAGS.agent_race,
        bot_race=FLAGS.bot_race,
        difficulty=FLAGS.difficulty,
        step_mul=FLAGS.step_mul,
        game_steps_per_episode=FLAGS.game_steps_per_episode,
        screen_size_px=(FLAGS.screen_resolution, FLAGS.screen_resolution),
        minimap_size_px=(FLAGS.minimap_resolution, FLAGS.minimap_resolution),
        visualize=visualize """) as env:
        env = available_actions_printer.AvailableActionsPrinter(env)
        agent = Agent()
        # run_loop([agent], env, FLAGS.max_agent_steps)
        agent.train(env, FLAGS.train)
        if FLAGS.save_replay:
            env.save_replay(Agent.__name__)

def main(unused_argv):
    """Run an agent."""
    if FLAGS.profile or FLAGS.trace:
        stopwatch.sw.enabled()
    else:
        stopwatch.sw.disable()
    
    if FLAGS.trace:
        stopwatch.sw.trace()

    maps.get(FLAGS.map)  # Assert the map exists.
    run_thread(FLAGS.map, FLAGS.render)

    if FLAGS.profile:
        print(stopwatch.sw)


def entry_point():  # Needed so setup.py scripts work.
    app.run(main)


if __name__ == "__main__":
    app.run(main)
