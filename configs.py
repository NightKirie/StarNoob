from absl import flags
from pysc2.lib import point_flag, features, actions
from pysc2.env import sc2_env

### Environment flags config

FLAGS = flags.FLAGS

#flags.DEFINE_enum("_only_use_kwargs", None, "Don't pass args, only kwargs")

flags.DEFINE_string("map_name", "Simple64", "Name of a SC2 map")

flags.DEFINE_bool("battle_net_map", False,
                  "Whether to use the battle.net versions of the map(s)")

flags.DEFINE_list("players", [sc2_env.Agent(sc2_env.Race.terran),
                              sc2_env.Bot(sc2_env.Race.terran,
                                          sc2_env.Difficulty.very_easy,
                                          sc2_env.BotBuild.macro)], "A List of Agent and Bot instances that specify who will play")

flags.DEFINE_float("discount", 1., "Returned as part of the observation")

flags.DEFINE_bool("discount_zero_after_timeout", False,
                  "If True, the discount will be zero after the 'game_steps_per_episode' timeout")

flags.DEFINE_bool("visualize", True,
                  "Whether to pop up a window showing the camera and feature layers.")

flags.DEFINE_integer("step_mul", 1024, """How many game steps per agent step (action/observation). 
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

# Load & save model or not
LOAD_MODEL = True
SAVE_MODEL = True

BUILDABLE_POSITION = [
    # (19, 23)
    [
        (13, 19), (13, 21), (13, 27), (15, 17), (15, 19), (15, 21), (15, 27), (17, 17), (17, 19), (17, 27),
        (17, 29), (21, 21), (21, 25), (21, 27),	(23, 17), (23, 19), (23, 21), (23, 25), (23, 27), (25, 17),
        (25, 19), (25, 21), (25, 25), (25, 27),	(27, 19), (27, 21), (27, 25), (27, 27), (29, 25), (19, 27),
        (19, 29), (13, 23), (15, 23), (17, 23), (21, 23), (23, 23), (25, 23), (27, 23), (29, 23),
    ],
    # (39, 45)
    [
        (29, 43), (31, 41), (31, 43), (31, 47), (31, 49), (31, 51), (33, 41), (33, 43), (33, 51), (35, 41),
        (35, 43), (35, 47), (35, 49), (37, 39), (37, 41), (41, 39), (41, 41), (41, 49), (41, 51), (43, 41),
        (43, 47), (43, 49), (43, 51), (45, 41), (45, 47), (45, 49), (39, 39), (39, 41), (39, 49), (39, 51),
        (29, 45), (31, 45), (33, 45), (35, 45), (43, 45), (45, 45),
    ],
    # (17, 48)
    [
        (11, 40), (13, 40), (15, 40), (17, 40), (19, 40), (21, 40), (23, 40), (25, 40), (11, 42), (13, 42),
        (15, 42), (17, 42), (19, 42), (21, 42), (23, 42), (9, 44), (11, 44), (13, 44), (15, 44), (17, 44),
        (19, 44), (21, 44), (23, 44), (9, 46), (21, 46), (23, 46), (9, 48), (11, 48), (13, 48), (21, 48),
        (23, 48), (9, 50), (11, 50), (13, 50), (21, 50), (23, 50), (9, 52), (11, 52), (13, 52), (15, 52),
        (17, 52), (21, 52), (23, 52), (11, 54), (13, 54), (15, 54), (17, 54), (21, 54), (23, 54)
    ],
    # (41, 21)
    [
        (33, 15), (35, 15), (41, 15), (43, 15), (45, 15), (47, 15), (33, 17), (35, 17), (41, 17), (43, 17),
        (45, 17), (47, 17), (35, 19), (37, 19), (45, 19), (47, 19), (35, 21), (37, 21), (35, 23), (37, 23),
        (35, 25), (37, 25), (39, 25), (41, 25), (43, 25), (45, 25), (47, 25), (33, 27), (35, 27), (37, 27),
        (39, 27), (41, 27), (43, 27), (45, 27), (47, 27), (33, 29), (35, 29), (37, 29), (39, 29), (41, 29),
        (43, 29), (45, 29),
    ],

]

MY_ARMY_LIST = [
    "Marine",
    "Reaper",
    "Marauder",
    "Ghost",
    "Hellion",
    "SiegeTank",
    #"WidowMine",
    "Hellbat",
    "Thor",
    "Liberator",
    "Cyclone",
    "VikingFighter",
    "Medivac",
    #"Raven",
    "Banshee",
    "Battlecruiser"]

MY_BUILDING_LIST = [
    "Armory",
    "Barracks",
    # "BarracksFlying",
    "BarracksReactor",
    "BarracksTechLab",
    # "Bunker",
    "CommandCenter",
    # "CommandCenterFlying",
    "EngineeringBay",
    "Factory",
    # "FactoryFlying",
    "FactoryReactor",
    "FactoryTechLab",
    "FusionCore",
    "GhostAcademy",
    # "MissileTurret",
    # "OrbitalCommand",
    # "OrbitalCommandFlying",
    # "PlanetaryFortress",
    # "Reactor",
    "Refinery",
    # "RefineryRich",
    # "SensorTower",
    "Starport",
    # "StarportFlying",
    "StarportReactor",
    "StarportTechLab",
    "SupplyDepot",
    # "SupplyDepotLowered",
    # "TechLab",
]

BUILD_BUILDING_CMD_ID = [
    183, # "Armory",
    185, # "Barracks",
    187, # "CommandCenter",
    191, # "EngineeringBay",
    194, # "Factory",
    195, # "FusionCore",
    196, # "GhostAcademy",
    214, # "Refinery",
    221, # "Starport",
    222, # "SupplyDepot",
]



ENEMY_ARMY_LIST = [
   "AutoTurret" ,
   "Banshee",
   "Battlecruiser",
   "Cyclone",
   "Ghost",
   "GhostAlternate",
   "GhostNova",
   "Hellion",
   "Hellbat",
   "KD8Charge",
   "Liberator",
   "LiberatorAG",
   "MULE",
   "Marauder",
   "Marine",
   "Medivac",
   "Nuke",
   "PointDefenseDrone",
   "Raven",
   "Reaper",
   "RepairDrone",
   "SCV",
   "SiegeTank",
   "SiegeTankSieged",
   "Thor",
   "ThorHighImpactMode",
   "VikingAssault",
   "VikingFighter",
   "WidowMine",
   "WidowMineBurrowed"
]

ENEMY_BUILDING_LIST = [
    "Armory",
    "Barracks",
    "BarracksFlying",
    "BarracksReactor",
    "BarracksTechLab",
    "Bunker",
    "CommandCenter",
    "CommandCenterFlying",
    "EngineeringBay",
    "Factory",
    "FactoryFlying",
    "FactoryReactor",
    "FactoryTechLab",
    "FusionCore",
    "GhostAcademy",
    "MissileTurret",
    "OrbitalCommand",
    "OrbitalCommandFlying",
    "PlanetaryFortress",
    "Reactor",
    "Refinery",
    "RefineryRich",
    "SensorTower",
    "Starport",
    "StarportFlying",
    "StarportReactor",
    "StarportTechLab",
    "SupplyDepot",
    "SupplyDepotLowered",
    "TechLab",
]

TRAINABLE_BUILDING = [
    "Barracks", 
    "Factory", 
    "Starport"
]

RESEARCH_NAME = [
    "TerranInfantryWeapons",
    "TerranInfantryArmor",
    "HiSecAutoTracking",
    "TerranStructureArmorUpgrade",
    "CombatShield",
    "Stimpack",
    "ConcussiveShells",
    "InfernalPreigniter",
    "CycloneLockOnDamage",
    "DrillingClaws",
    "SmartServos",
    "RavenCorvidReactor",
    "BansheeCloakingField",
    "BansheeHyperflightRotors",
    "TerranVehicleWeapons", 
    "TerranShipWeapons", 
    "TerranVehicleAndShipPlating"
]

MAIN_COMMAND_CENTER_POTISION = [
    {'x': 39, 'y': 45}, 
    {'x': 19, 'y': 23}
]