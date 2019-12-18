from absl import flags
from pysc2.lib import point_flag, features, actions
from pysc2.env import sc2_env

FLAGS = flags.FLAGS

#flags.DEFINE_enum("_only_use_kwargs", None, "Don't pass args, only kwargs")

flags.DEFINE_string("map_name", "Simple64", "Name of a SC2 map")

flags.DEFINE_bool("battle_net_map", False,
                  "Whether to use the battle.net versions of the map(s)")

flags.DEFINE_list("players", [sc2_env.Agent(sc2_env.Race.terran),
                              sc2_env.Bot(sc2_env.Race.terran,
                                          sc2_env.Difficulty.very_easy)], "A List of Agent and Bot instances that specify who will play")

flags.DEFINE_float("discount", 1., "Returned as part of the observation")

flags.DEFINE_bool("discount_zero_after_timeout", False,
                  "If True, the discount will be zero after the 'game_steps_per_episode' timeout")

flags.DEFINE_bool("visualize", False,
                  "Whether to pop up a window showing the camera and feature layers.")

flags.DEFINE_integer("step_mul", None, """How many game steps per agent step (action/observation). 
                                            None means use the map default.""")

flags.DEFINE_bool("realtime", False, """Whether to use realtime mode. In this mode the game simulation
                                        automatically advances (at 22.4 gameloops per second) rather than
                                        being stepped manually.""")

flags.DEFINE_integer("save_replay_episodes", 0, """Save a replay after this many episodes. Default of 0
                                                means don't save replays.""")

flags.DEFINE_string("replay_dir", None,
                    "Directory to save replays. Required with save_replay_episodes.")

flags.DEFINE_string("replay_prefix", None,
                    "An optional prefix to use when saving replays.")

flags.DEFINE_integer("game_steps_per_episode", None, """Game steps per episode, independent of the
                                                        step_mul. 0 means no limit. None means use the map default.""")

flags.DEFINE_integer("score_index", None, """-1 means use the win/loss reward, >=0 is the index into the
                                            score_cumulative with 0 being the curriculum score. None means use
                                            the map default.""")

flags.DEFINE_integer("score_multiplier", None,
                     "How much to multiply the score by. Useful for negating.")

flags.DEFINE_integer("random_seed", None, """Random number seed to use when initializing the game. This
                                            lets you run repeatable games/tests.""")

flags.DEFINE_bool("disable_fog", False, "Whether to disable fog of war.")

flags.DEFINE_bool("ensure_available_actions", False, """Whether to throw an exception when an
                                                        unavailable action is passed to step().""")

flags.DEFINE_string("version", None, "The version of SC2 to use, defaults to the latest.")


# For agent_interface_format
point_flag.DEFINE_point("feature_screen_size", "64",
                            "Resolution for screen feature layers.")
point_flag.DEFINE_point("feature_minimap_size", "64",
                        "Resolution for minimap feature layers.")
point_flag.DEFINE_point("rgb_screen_size", None,
                        "Resolution for rendered screen.")
point_flag.DEFINE_point("rgb_minimap_size", None,
                        "Resolution for rendered minstring("")imap.")
flags.DEFINE_bool("use_raw_units", True,
                  "Whether to include raw unit data in observations.")
flags.DEFINE_bool("use_raw_actions", True, 
                  "Whether to use raw actions as the interface. Same as specifying action_space=ActionSpace.RAW")
flags.DEFINE_integer("raw_resolution", 64, 
                     "Discretize the `raw_units` observation's x,y to this resolution. Default is the map_size")                   