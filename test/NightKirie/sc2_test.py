from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
import sys
from absl import flags
FLAGS = flags.FLAGS
FLAGS(sys.argv)

players = []

players.append(sc2_env.Agent(sc2_env.Race["terran"]))
players.append(sc2_env.Bot(
    sc2_env.Race["terran"], sc2_env.Difficulty["very_easy"]))


env = sc2_env.SC2Env(players=players,
                     map_name="Simple64",
                     agent_interface_format=features.AgentInterfaceFormat(
                         action_space=actions.ActionSpace.RAW,
                         use_raw_units=True,
                         raw_resolution=64)
                     )

print(type(env._replay_dir))
