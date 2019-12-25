from base_agent import *

DATA_FILE = 'Sub_building_data'
FAILED_COMMAND = 0.0001
MORE_MINERALS_USED_REWARD_RATE = 0.0001
MORE_VESPENE_USED_REWARD_RATE = 0.0002

BATCH_SIZE = 128
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 500

SAVE_POLICY_NET = 'model/economic_dqn_policy'
SAVE_TARGET_NET = 'model/economic_dqn_target'
SAVE_MEMORY = 'model/economic_memory'

class Agent(BaseAgent):

    actions = ("do_nothing",
               "harvest_minerals",
               "build_supply_depot",
               "build_barracks",
               "build_refinery",
               "train_SCV",
               "harvest_gas",
               "build_barrack_techlab",
               "build_barrack_reactor",
               "build_ghostacademys",
               "build_engineeringbays",
               "build_factorys",
               "build_factory_techlab",
               "build_factory_reactor",
               "build_armorys",
               "build_starports",
               "build_starports_techlab",
               "build_starports_reactor",
               "build_fusioncores",
               "research_infantryweapons",
               "research_infantryarmor",
               "research_hiSecautotracking",
               "research_structurearmor",
               "research_Combat_Shield",
               "research_StimPack",
               "research_ConcussiveShells",
               "research_InfernalPreignite",
               "research_CycloneLockOnDamage",
               "research_DrillingClaws",
               "research_SmartServos",
               "research_RavenCorvidReactor",
               "research_BansheeCloakingField",
               "research_BansheeHyperflightRotors",
               "research_TerranVehicleWeapons",
               "research_TerranShipWeapons",
               "research_TerranVehicleAndShipPlating"
               )

    def __init__(self):
        super(Agent, self).__init__()

    def get_distances(self, obs, units, xy):
        units_xy = [(unit.x, unit.y) for unit in units]
        return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1)

    def get_least_busy_building(self, building):
        order_array = []
        for item in building:
            order_array.append(item.order_length)
        return order_array.index(min(order_array))

    def get_notfull_worker_building_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.build_progress == 100
                and unit.assigned_harvesters < unit.ideal_harvesters
                and unit.alliance == features.PlayerRelative.SELF]

    def harvest_minerals(self, obs):
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        idle_scvs = [scv for scv in scvs if scv.order_length == 0]
        if len(idle_scvs) > 0:
            mineral_patches = [unit for unit in obs.observation.raw_units
                               if unit.unit_type in [
                                   units.Neutral.BattleStationMineralField,
                                   units.Neutral.BattleStationMineralField750,
                                   units.Neutral.LabMineralField,
                                   units.Neutral.LabMineralField750,
                                   units.Neutral.MineralField,
                                   units.Neutral.MineralField750,
                                   units.Neutral.PurifierMineralField,
                                   units.Neutral.PurifierMineralField750,
                                   units.Neutral.PurifierRichMineralField,
                                   units.Neutral.PurifierRichMineralField750,
                                   units.Neutral.RichMineralField,
                                   units.Neutral.RichMineralField750
                               ]]
            scv = random.choice(idle_scvs)
            distances = self.get_distances(
                obs, mineral_patches, (scv.x, scv.y))
            mineral_patch = mineral_patches[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Harvest_Gather_unit(
                "now", scv.tag, mineral_patch.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def harvest_gas(self, obs):
        notfull_completed_refinerys = self.get_notfull_worker_building_by_type(
            obs, units.Terran.Refinery)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        if len(scvs) > 0 and len(notfull_completed_refinerys) > 0:
            scv = random.choice(scvs)
            distances = self.get_distances(
                obs, notfull_completed_refinerys, (scv.x, scv.y))
            refinery = notfull_completed_refinerys[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Harvest_Gather_unit(
                "now", scv.tag, refinery.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def train_SCV(self, obs):
        completed_commandcenters = self.get_my_completed_units_by_type(
            obs, units.Terran.CommandCenter)
        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)

        if len(completed_commandcenters) > 0 and free_supply > 0:
            completed_commandcenter = completed_commandcenters[self.get_least_busy_building(
                completed_commandcenters)]
            return actions.RAW_FUNCTIONS.Train_SCV_quick("now", completed_commandcenter.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def build_supply_depot(self, obs):
        supply_depots = self.get_my_units_by_type(
            obs, units.Terran.SupplyDepot)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        if len(supply_depots) >= 9:
            return actions.RAW_FUNCTIONS.no_op()
        else:
            if (len(supply_depots) == 0 and len(scvs) > 0):
                supply_depot_xy = (22, 26) if self.base_top_left else (35, 42)
            elif (len(supply_depots) == 1 and len(scvs) > 0):
                supply_depot_xy = (21, 26) if self.base_top_left else (36, 42)
            elif (len(supply_depots) == 2 and len(scvs) > 0):
                supply_depot_xy = (19, 26) if self.base_top_left else (38, 42)
            elif (len(supply_depots) == 3 and len(scvs) > 0):
                supply_depot_xy = (17, 26) if self.base_top_left else (40, 42)
            elif (len(supply_depots) == 4 and len(scvs) > 0):
                supply_depot_xy = (12, 18) if self.base_top_left else (46, 42)
            elif (len(supply_depots) == 5 and len(scvs) > 0):
                supply_depot_xy = (12, 20) if self.base_top_left else (46, 44)
            elif (len(supply_depots) == 6 and len(scvs) > 0):
                supply_depot_xy = (12, 22) if self.base_top_left else (46, 46)
            elif (len(supply_depots) == 7 and len(scvs) > 0):
                supply_depot_xy = (12, 24) if self.base_top_left else (46, 48)
            elif (len(supply_depots) == 8 and len(scvs) > 0):
                supply_depot_xy = (12, 26) if self.base_top_left else (45, 48)
            else:
                return actions.RAW_FUNCTIONS.no_op()
            distances = self.get_distances(obs, scvs, supply_depot_xy)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt(
                "now", scv.tag, supply_depot_xy)
        return actions.RAW_FUNCTIONS.no_op()

    def build_refinery(self, obs):
        refinery = self.get_my_units_by_type(obs, units.Terran.Refinery)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        gas_patches = [unit for unit in obs.observation.raw_units
                       if unit.unit_type in [
                           units.Neutral.ProtossVespeneGeyser,
                           units.Neutral.PurifierVespeneGeyser,
                           units.Neutral.RichVespeneGeyser,
                           units.Neutral.ShakurasVespeneGeyser,
                           units.Neutral.SpacePlatformGeyser,
                           units.Neutral.VespeneGeyser
                       ]]
        if (len(refinery) == 0 and len(scvs) > 0):
            scv = random.choice(scvs)
            distances = self.get_distances(obs, gas_patches, (scv.x, scv.y))
            gas_patch = gas_patches[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_Refinery_pt("now", scv.tag, gas_patch.tag)
        if (len(refinery) == 1 and len(scvs) > 0):
            scv = random.choice(scvs)
            distances = self.get_distances(obs, gas_patches, (scv.x, scv.y))
            gas_patch = gas_patches[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_Refinery_pt("now", scv.tag, gas_patch.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def build_barracks(self, obs):
        completed_supply_depots = self.get_my_completed_units_by_type(
            obs, units.Terran.SupplyDepot)
        barrackses = self.get_my_units_by_type(obs, units.Terran.Barracks)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        if (len(completed_supply_depots) > 0 and len(barrackses) == 0 and
                obs.observation.player.minerals >= 150 and len(scvs) > 0):
            barracks_xy = (22, 21) if self.base_top_left else (35, 45)
            distances = self.get_distances(obs, scvs, barracks_xy)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_Barracks_pt(
                "now", scv.tag, barracks_xy)
        if (len(completed_supply_depots) > 0 and len(barrackses) == 1 and
                obs.observation.player.minerals >= 150 and len(scvs) > 0):
            barracks_xy = (22, 24) if self.base_top_left else (35, 47)
            distances = self.get_distances(obs, scvs, barracks_xy)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_Barracks_pt(
                "now", scv.tag, barracks_xy)
        return actions.RAW_FUNCTIONS.no_op()

    def build_barrack_techlab(self, obs):
        completed_barrackses = self.get_my_completed_units_by_type(
            obs, units.Terran.Barracks)
        if len(completed_barrackses) > 0:
            return actions.RAW_FUNCTIONS.Build_TechLab_Barracks_quick(
                "now", [barrack.tag for barrack in completed_barrackses])
        return actions.RAW_FUNCTIONS.no_op()

    def build_barrack_reactor(self, obs):
        completed_barrackses = self.get_my_completed_units_by_type(
            obs, units.Terran.Barracks)
        if len(completed_barrackses) > 0:
            return actions.RAW_FUNCTIONS.Build_Reactor_Barracks_quick(
                "now", [barrack.tag for barrack in completed_barrackses])
        return actions.RAW_FUNCTIONS.no_op()

    def build_ghostacademys(self, obs):
        ghostacademys = self.get_my_units_by_type(obs, units.Terran.GhostAcademy)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        if (len(ghostacademys) == 0 and obs.observation.player.minerals >= 150 and len(scvs) > 0):
            ghostacademys_xy = (25, 17) if self.base_top_left else (35, 49)
            distances = self.get_distances(obs, scvs, ghostacademys_xy)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_GhostAcademy_pt(
                "now", scv.tag, ghostacademys_xy)
        return actions.RAW_FUNCTIONS.no_op()

    def build_engineeringbays(self, obs):
        engineeringbays = self.get_my_units_by_type(
            obs, units.Terran.EngineeringBay)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        if len(engineeringbays) == 0 and len(scvs) > 0:
            engineeringbays_xy = (19, 28) if self.base_top_left else (38, 40)
            distances = self.get_distances(obs, scvs, engineeringbays_xy)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_EngineeringBay_pt(
                "now", scv.tag, engineeringbays_xy)
        return actions.RAW_FUNCTIONS.no_op()

    def build_factorys(self, obs):
        factorys = self.get_my_units_by_type(
            obs, units.Terran.Factory)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        if len(factorys) == 0 and len(scvs) > 0:
            factorys_xy = (21, 28) if self.base_top_left else (43, 41)
            distances = self.get_distances(obs, scvs, factorys_xy)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_Factory_pt(
                "now", scv.tag, factorys_xy)
        return actions.RAW_FUNCTIONS.no_op()

    def build_factory_techlab(self, obs):
        completed_factorys = self.get_my_completed_units_by_type(
            obs, units.Terran.Factory)
        if len(completed_factorys) > 0:
            return actions.RAW_FUNCTIONS.Build_TechLab_Factory_quick(
                "now", [factorys.tag for factorys in completed_factorys])
        return actions.RAW_FUNCTIONS.no_op()

    def build_factory_reactor(self, obs):
        completed_factorys = self.get_my_completed_units_by_type(
            obs, units.Terran.Factory)
        if len(completed_factorys) > 0:
            return actions.RAW_FUNCTIONS.Build_Reactor_Factory_quick(
                "now", [factorys.tag for factorys in completed_factorys])
        return actions.RAW_FUNCTIONS.no_op()

    def build_armorys(self, obs):
        armorys = self.get_my_units_by_type(
            obs, units.Terran.Armory)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        if len(armorys) == 0 and len(scvs) > 0:
            armorys_xy = (17, 28) if self.base_top_left else (40, 40)
            distances = self.get_distances(obs, scvs, armorys_xy)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_Armory_pt(
                "now", scv.tag, armorys_xy)
        return actions.RAW_FUNCTIONS.no_op()

    def build_starports(self, obs):
        starports = self.get_my_units_by_type(
            obs, units.Terran.Starport)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        if len(starports) == 0 and len(scvs) > 0:
            starports_xy = (25, 21) if self.base_top_left else (32, 45)
            distances = self.get_distances(obs, scvs, starports_xy)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_Starport_pt(
                "now", scv.tag, starports_xy)
        return actions.RAW_FUNCTIONS.no_op()

    def build_fusioncores(self, obs):
        fusioncores = self.get_my_units_by_type(
            obs, units.Terran.FusionCore)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        if len(fusioncores) == 0 and len(scvs) > 0:
            fusioncores_xy = (25, 19) if self.base_top_left else (32, 47)
            distances = self.get_distances(obs, scvs, fusioncores_xy)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_FusionCore_pt(
                "now", scv.tag, fusioncores_xy)
        return actions.RAW_FUNCTIONS.no_op()

    def build_starports_techlab(self, obs):
        completed_starports = self.get_my_completed_units_by_type(
            obs, units.Terran.Starport)
        if len(completed_starports) > 0:
            return actions.RAW_FUNCTIONS.Build_TechLab_Starport_quick(
                "now", [starports.tag for starports in completed_starports])
        return actions.RAW_FUNCTIONS.no_op()

    def build_starports_reactor(self, obs):
        completed_starports = self.get_my_completed_units_by_type(
            obs, units.Terran.Starport)
        if len(completed_starports) > 0:
            return actions.RAW_FUNCTIONS.Build_Reactor_Starport_quick(
                "now", [starports.tag for starports in completed_starports])
        return actions.RAW_FUNCTIONS.no_op()

    def research_infantryweapons(self, obs):
        completed_engineeringbays = self.get_my_completed_units_by_type(
            obs, units.Terran.EngineeringBay)
        if len(completed_engineeringbays) > 0:
            return actions.RAW_FUNCTIONS.Research_TerranInfantryWeapons_quick(
                "now", [engineeringbay.tag for engineeringbay in completed_engineeringbays])
        return actions.RAW_FUNCTIONS.no_op()

    def research_infantryarmor(self, obs):
        completed_engineeringbays = self.get_my_completed_units_by_type(
            obs, units.Terran.EngineeringBay)
        if len(completed_engineeringbays) > 0:
            return actions.RAW_FUNCTIONS.Research_TerranInfantryArmor_quick(
                "now", [engineeringbay.tag for engineeringbay in completed_engineeringbays])
        return actions.RAW_FUNCTIONS.no_op()

    def research_hiSecautotracking(self, obs):
        completed_engineeringbays = self.get_my_completed_units_by_type(
            obs, units.Terran.EngineeringBay)
        if len(completed_engineeringbays) > 0:
            return actions.RAW_FUNCTIONS.Research_HiSecAutoTracking_quick(
                "now", [engineeringbay.tag for engineeringbay in completed_engineeringbays])
        return actions.RAW_FUNCTIONS.no_op()

    def research_structurearmor(self, obs):
        completed_engineeringbays = self.get_my_completed_units_by_type(
            obs, units.Terran.EngineeringBay)
        if len(completed_engineeringbays) > 0:
            return actions.RAW_FUNCTIONS.Research_TerranStructureArmorUpgrade_quick(
                "now", [engineeringbay.tag for engineeringbay in completed_engineeringbays])
        return actions.RAW_FUNCTIONS.no_op()
#Tech lab (barrack)
    def research_Combat_Shield(self, obs):
        completed_Tech_lab = self.get_my_completed_units_by_type(
            obs, units.Terran.BarracksTechLab)
        if len(completed_Tech_lab) > 0:
            return actions.RAW_FUNCTIONS.Research_CombatShield_quick(
                "now", [Tech_lab.tag for Tech_lab in completed_Tech_lab])
        return actions.RAW_FUNCTIONS.no_op()

    def research_StimPack(self, obs):
        completed_Tech_lab = self.get_my_completed_units_by_type(
            obs, units.Terran.BarracksTechLab)
        if len(completed_Tech_lab) > 0:
            return actions.RAW_FUNCTIONS.Research_Stimpack_quick(
                "now", [Tech_lab.tag for Tech_lab in completed_Tech_lab])
        return actions.RAW_FUNCTIONS.no_op()

    def research_ConcussiveShells(self, obs):
        completed_Tech_lab = self.get_my_completed_units_by_type(
            obs, units.Terran.BarracksTechLab)
        if len(completed_Tech_lab) > 0:
            return actions.RAW_FUNCTIONS.Research_ConcussiveShells_quick(
                "now", [Tech_lab.tag for Tech_lab in completed_Tech_lab])
        return actions.RAW_FUNCTIONS.no_op()

#Tech lab (factory)
    def research_InfernalPreignite(self, obs):
        completed_Tech_lab = self.get_my_completed_units_by_type(
            obs, units.Terran.FactoryTechLab)
        if len(completed_Tech_lab) > 0:
            return actions.RAW_FUNCTIONS.Research_InfernalPreigniter_quick(
                "now", [Tech_lab.tag for Tech_lab in completed_Tech_lab])
        return actions.RAW_FUNCTIONS.no_op()

    def research_CycloneLockOnDamage(self, obs):
        completed_Tech_lab = self.get_my_completed_units_by_type(
            obs, units.Terran.FactoryTechLab)
        if len(completed_Tech_lab) > 0:
            return actions.RAW_FUNCTIONS.Research_CycloneLockOnDamage_quick(
                "now", [Tech_lab.tag for Tech_lab in completed_Tech_lab])
        return actions.RAW_FUNCTIONS.no_op()

    def research_DrillingClaws(self, obs):
        completed_Tech_lab = self.get_my_completed_units_by_type(
            obs, units.Terran.FactoryTechLab)
        if len(completed_Tech_lab) > 0:
            return actions.RAW_FUNCTIONS.Research_DrillingClaws_quick(
                "now", [Tech_lab.tag for Tech_lab in completed_Tech_lab])
        return actions.RAW_FUNCTIONS.no_op()

    def research_SmartServos(self, obs):
        completed_Tech_lab = self.get_my_completed_units_by_type(
            obs, units.Terran.FactoryTechLab)
        if len(completed_Tech_lab) > 0:
            return actions.RAW_FUNCTIONS.Research_SmartServos_quick(
                "now", [Tech_lab.tag for Tech_lab in completed_Tech_lab])
        return actions.RAW_FUNCTIONS.no_op()

#Tech lab (starport)

    def research_RavenCorvidReactor(self, obs):
        completed_Tech_lab = self.get_my_completed_units_by_type(
            obs, units.Terran.StarportTechLab)
        if len(completed_Tech_lab) > 0:
            return actions.RAW_FUNCTIONS.Research_RavenCorvidReactor_quick(
                "now", [Tech_lab.tag for Tech_lab in completed_Tech_lab])
        return actions.RAW_FUNCTIONS.no_op()

    def research_BansheeCloakingField(self, obs):
        completed_Tech_lab = self.get_my_completed_units_by_type(
            obs, units.Terran.StarportTechLab)
        if len(completed_Tech_lab) > 0:
            return actions.RAW_FUNCTIONS.Research_BansheeCloakingField_quick(
                "now", [Tech_lab.tag for Tech_lab in completed_Tech_lab])
        return actions.RAW_FUNCTIONS.no_op()

    def research_BansheeHyperflightRotors(self, obs):
        completed_Tech_lab = self.get_my_completed_units_by_type(
            obs, units.Terran.StarportTechLab)
        if len(completed_Tech_lab) > 0:
            return actions.RAW_FUNCTIONS.Research_BansheeHyperflightRotors_quick(
                "now", [Tech_lab.tag for Tech_lab in completed_Tech_lab])
        return actions.RAW_FUNCTIONS.no_op()

#Armory
    def research_TerranVehicleWeapons(self, obs):
        completed_armory = self.get_my_completed_units_by_type(
            obs, units.Terran.Armory)
        if len(completed_armory) > 0:
            return actions.RAW_FUNCTIONS.Research_TerranVehicleWeapons_quick(
                "now", [armory.tag for armory in completed_armory])
        return actions.RAW_FUNCTIONS.no_op()

    def research_TerranShipWeapons(self, obs):
        completed_armory = self.get_my_completed_units_by_type(
            obs, units.Terran.Armory)
        if len(completed_armory) > 0:
            return actions.RAW_FUNCTIONS.Research_TerranShipWeapons_quick(
                "now", [armory.tag for armory in completed_armory])
        return actions.RAW_FUNCTIONS.no_op()

    def research_TerranVehicleAndShipPlating(self, obs):
        completed_armory = self.get_my_completed_units_by_type(
            obs, units.Terran.Armory)
        if len(completed_armory) > 0:
            return actions.RAW_FUNCTIONS.Research_TerranVehicleAndShipPlating_quick(
                "now", [armory.tag for armory in completed_armory])
        return actions.RAW_FUNCTIONS.no_op()


class SubAgent_Economic(Agent):

    def __init__(self):
        super(SubAgent_Economic, self).__init__()
        self.new_game()
        self.state_size = len(self.get_state(MYOBS))
        self.action_size = len(self.actions)
        self.policy_net = DQN(self.state_size, self.action_size, SAVE_POLICY_NET).to(device)
        self.target_net = DQN(self.state_size, self.action_size, SAVE_TARGET_NET).to(device)

        self.memory = ReplayMemory(10000)

        # if saved models exist
        if self.policy_net.load() and self.target_net.load():
            with open(SAVE_MEMORY, 'rb') as f:
                self.memory = pickle.load(f)
                log.log(LOG_MODEL, "Load memory " + SAVE_MEMORY)
        else:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
            log.log(LOG_MODEL, "Memory " + SAVE_MEMORY + " not found")

        self.optimizer = optim.RMSprop(self.policy_net.parameters())

        self.episode = 0

    def reset(self):
        log.debug('in reset')
        super(SubAgent_Economic, self).reset()
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
        self.negative_reward = 0

    def get_negative_reward(self, obs, action):

        player_mineral = obs.observation.player.minerals
        player_vespene = obs.observation.player.vespene
        free_supply = (obs.observation.player.food_cap - obs.observation.player.food_used)

        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        idle_scvs = [scv for scv in scvs if scv.order_length == 0]

        refinery = self.get_my_units_by_type(obs, units.Terran.Refinery)
        notfull_completed_refinerys = self.get_notfull_worker_building_by_type(obs, units.Terran.Refinery)

        completed_commandcenters = self.get_my_completed_units_by_type(obs, units.Terran.CommandCenter)

        supply_depots = self.get_my_units_by_type(obs, units.Terran.SupplyDepot)
        completed_supply_depots = self.get_my_completed_units_by_type(obs, units.Terran.SupplyDepot)

        barrackses = self.get_my_units_by_type(obs, units.Terran.Barracks)
        completed_barrackses = self.get_my_completed_units_by_type(obs, units.Terran.Barracks)
        barrackstechlabs = self.get_my_units_by_type(obs, units.Terran.BarracksTechLab)
        barracksreactors = self.get_my_units_by_type(obs, units.Terran.BarracksReactor)
        completed_barrackstechlabs = self.get_my_completed_units_by_type(obs, units.Terran.BarracksTechLab)

        ghostacademys = self.get_my_units_by_type(obs, units.Terran.GhostAcademy)

        engineeringbays = self.get_my_units_by_type(obs, units.Terran.EngineeringBay)
        completed_engineeringbays = self.get_my_completed_units_by_type(obs, units.Terran.EngineeringBay)

        factorys = self.get_my_units_by_type(obs, units.Terran.Factory)
        completed_factorys = self.get_my_completed_units_by_type(obs, units.Terran.Factory)
        factorytechlabs = self.get_my_units_by_type(obs, units.Terran.FactoryTechLab)
        factoryreactors = self.get_my_units_by_type(obs, units.Terran.FactoryReactor)
        completed_factorytechlabs = self.get_my_completed_units_by_type(obs, units.Terran.FactoryTechLab)

        armorys = self.get_my_units_by_type(obs, units.Terran.Armory)
        completed_armorys = self.get_my_completed_units_by_type(obs, units.Terran.Armory)

        starports = self.get_my_units_by_type(obs, units.Terran.Starport)
        complete_starports = self.get_my_completed_units_by_type(obs, units.Terran.Starport)
        starportstechlabs = self.get_my_units_by_type(obs, units.Terran.StarportTechLab)
        starportsreactors = self.get_my_units_by_type(obs, units.Terran.StarportReactor)
        completed_starportstechlabs = self.get_my_completed_units_by_type(obs, units.Terran.StarportTechLab )

        fusioncores = self.get_my_units_by_type(obs, units.Terran.FusionCore)

        have_research_infantryweapons_level1 = 7 in obs.observation.upgrades
        have_research_infantryweapons_level2 = 8 in obs.observation.upgrades
        have_research_infantryweapons_level3 = 9 in obs.observation.upgrades
        have_research_infantryarmor_level1 = 11 in obs.observation.upgrades
        have_research_infantryarmor_level2 = 12 in obs.observation.upgrades
        have_research_infantryarmor_level3 = 13 in obs.observation.upgrades
        have_research_hiSecautotracking = 5 in obs.observation.upgrades
        have_research_structurearmor = 6 in obs.observation.upgrades

        have_research_Combat_Shield = 16 in obs.observation.upgrades
        have_research_StimPack = 15 in obs.observation.upgrades
        have_research_ConcussiveShells = 17 in obs.observation.upgrades

        have_research_InfernalPreignite = 19 in obs.observation.upgrades
        have_research_CycloneLockOnDamage = 144 in obs.observation.upgrades
        have_research_DrillingClaws = 122 in obs.observation.upgrades
        have_research_SmartServos = 289 in obs.observation.upgrades

        have_research_RavenCorvidReactor = 22 in obs.observation.upgrades
        have_research_BansheeCloakingField = 20 in obs.observation.upgrades
        have_research_BansheeHyperflightRotors = 136 in obs.observation.upgrades

        have_research_TerranVehicleWeapons_level_1 = 30 in obs.observation.upgrades
        have_research_TerranVehicleWeapons_level_2 = 31 in obs.observation.upgrades
        have_research_TerranVehicleWeapons_level_3 = 32 in obs.observation.upgrades

        have_research_TerranShipWeapons_level_1 = 36 in obs.observation.upgrades
        have_research_TerranShipWeapons_level_2 = 37 in obs.observation.upgrades
        have_research_TerranShipWeapons_level_3 = 38 in obs.observation.upgrades

        have_research_TerranVehicleAndShipPlating_level_1 = 116 in obs.observation.upgrades
        have_research_TerranVehicleAndShipPlating_level_2 = 117 in obs.observation.upgrades
        have_research_TerranVehicleAndShipPlating_level_3 = 118 in obs.observation.upgrades


        if action == "harvest_minerals":
            if len(idle_scvs) == 0:
                return FAILED_COMMAND
        elif action == "build_supply_depot":
            if len(supply_depots) >= 9 or len(scvs) <= 0 or player_mineral < 100:
                return FAILED_COMMAND
        elif action == "build_barracks":
            if len(completed_supply_depots) <= 0 or len(barrackses) >= 2 or len(scvs) <= 0 or player_mineral < 150:
                return FAILED_COMMAND
        elif action == "build_refinery":
            if len(refinery) >= 2 or len(scvs) <= 0 or player_mineral < 75:
                return FAILED_COMMAND
        elif action == "train_SCV":
            if len(completed_commandcenters) <= 0 or free_supply == 0 or player_mineral < 50:
                return FAILED_COMMAND
        elif action == "harvest_gas":
            if len(scvs) <= 0 or len(notfull_completed_refinerys) <= 0:
                return FAILED_COMMAND
        elif action == "build_barrack_techlab":
            if len(completed_barrackses) <= 0 or len(completed_barrackses) <= len(barracksreactors) + len(barrackstechlabs) or player_mineral < 50 or player_vespene < 25:
                return FAILED_COMMAND
        elif action == "build_barrack_reactor":
            if len(completed_barrackses) <= 0 or len(completed_barrackses) <= len(barracksreactors) + len(barrackstechlabs) or player_mineral < 50 or player_vespene < 50:
                return FAILED_COMMAND
        elif action == "build_ghostacademys":
            if len(ghostacademys) >= 1 or len(completed_barrackses) <= 0 or len(scvs) <= 0 or player_mineral < 150 or player_vespene < 50:
                return FAILED_COMMAND
        elif action == "build_engineeringbays":
            if len(engineeringbays) >= 1 or len(scvs) <= 0 or len(completed_commandcenters) <= 0 or player_mineral < 125:
                return FAILED_COMMAND
        elif action == "build_factorys":
            if len(factorys) >= 1 or len(completed_barrackses) <= 0 or len(scvs) <= 0 or player_mineral < 150 or player_vespene < 100:
                return FAILED_COMMAND
        elif action == "build_factory_techlab":
            if len(completed_factorys) <= 0 or len(completed_factorys) <= len(factoryreactors) + len(factorytechlabs) or player_mineral < 50 or player_vespene < 25:
                return FAILED_COMMAND
        elif action == "build_factory_reactor":
            if len(completed_factorys) <= 0 or len(completed_factorys) <= len(factoryreactors) + len(factorytechlabs) or player_mineral < 50 or player_vespene < 50:
                return FAILED_COMMAND
        elif action == "build_armorys":
            if len(armorys) >= 1 or len(completed_factorys) <= 0 or len(scvs) <= 0 or player_mineral < 150 or player_vespene < 100:
                return FAILED_COMMAND
        elif action == "build_starports":
            if len(starports) >= 1 or len(completed_factorys) <= 0 or len(scvs) <= 0 or player_mineral < 150 or player_vespene <100:
                return FAILED_COMMAND
        elif action == "build_starports_techlab":
            if len(complete_starports) <= 0 or len(complete_starports) <= len(starportsreactors) + len(starportstechlabs) or player_mineral < 50 or player_vespene < 25:
                return FAILED_COMMAND
        elif action == "build_starports_reactor":
            if len(complete_starports) <= 0 or len(complete_starports) <= len(starportsreactors) + len(starportstechlabs) or player_mineral < 50 or player_vespene < 50:
                return FAILED_COMMAND
        elif action == "build_fusioncores":
            if len(fusioncores) >= 1 or len(complete_starports) <= 0 or len(scvs) <= 0 or player_mineral < 150 or player_vespene < 150:
                return FAILED_COMMAND
        elif action == "research_infantryweapons":
            if len(completed_engineeringbays) <= 0 or have_research_infantryweapons_level3:
                return FAILED_COMMAND
            if have_research_infantryweapons_level2:
                if len(completed_armorys) <= 0 or player_mineral < 250 or player_vespene < 250:
                    return FAILED_COMMAND
            if have_research_infantryweapons_level1:
                if len(completed_armorys) <= 0 or player_mineral < 175 or player_vespene < 175:
                    return FAILED_COMMAND
            if player_mineral < 100 or player_vespene < 100:
                return FAILED_COMMAND
        elif action == "research_infantryarmor":
            if len(completed_engineeringbays) <= 0 or have_research_infantryarmor_level3:
                return FAILED_COMMAND
            if have_research_infantryarmor_level2:
                if len(completed_armorys) <= 0 or player_mineral < 250 or player_vespene < 250:
                    return FAILED_COMMAND
            if have_research_infantryarmor_level1:
                if len(completed_armorys) <= 0 or player_mineral < 175 or player_vespene < 175:
                    return FAILED_COMMAND
            if player_mineral < 100 or player_vespene < 100:
                return FAILED_COMMAND
        elif action == "research_hiSecautotracking":
            if len(completed_engineeringbays) <= 0 or have_research_hiSecautotracking or player_mineral < 100 or player_vespene < 100:
                return FAILED_COMMAND
        elif action == "research_structurearmor":
            if len(completed_engineeringbays) <= 0 or have_research_structurearmor or player_mineral < 100 or player_vespene < 100:
                return FAILED_COMMAND
        elif action == "research_Combat_Shield":
            if len(completed_barrackstechlabs) <= 0 or have_research_Combat_Shield or player_mineral < 100 or player_vespene < 100:
                return FAILED_COMMAND
        elif action == "research_StimPack":
            if len(completed_barrackstechlabs) <= 0 or have_research_StimPack or player_mineral < 100 or player_vespene < 100:
                return FAILED_COMMAND
        elif action == "research_ConcussiveShells":
            if len(completed_barrackstechlabs) <= 0 or have_research_ConcussiveShells or player_mineral < 50 or player_vespene < 50:
                return FAILED_COMMAND
        elif action == "research_InfernalPreignite":
            if len(completed_factorytechlabs) <= 0 or have_research_InfernalPreignite or player_mineral < 100 or player_vespene < 100:
                return FAILED_COMMAND
        elif action == "research_CycloneLockOnDamage":
            if len(completed_factorytechlabs) <= 0 or have_research_CycloneLockOnDamage or player_mineral < 100 or player_vespene < 100:
                return FAILED_COMMAND
        elif action == "research_DrillingClaws":
            if len(completed_factorytechlabs) <= 0 or have_research_DrillingClaws or player_mineral < 75 or player_vespene < 75:
                return FAILED_COMMAND
        elif action == "research_SmartServos":
            if len(completed_factorytechlabs) <= 0 or have_research_SmartServos or player_mineral < 100 or player_vespene < 100:
                return FAILED_COMMAND
        elif action == "research_RavenCorvidReactor":
            if len(completed_starportstechlabs) <= 0 or have_research_RavenCorvidReactor or player_mineral < 150 or player_vespene < 150:
                return FAILED_COMMAND
        elif action == "research_BansheeCloakingField":
            if len(completed_starportstechlabs) <= 0 or have_research_BansheeCloakingField or player_mineral < 100 or player_vespene < 100:
                return FAILED_COMMAND
        elif action == "research_BansheeHyperflightRotors":
            if len(completed_starportstechlabs) <= 0 or have_research_BansheeHyperflightRotors or player_mineral < 150 or player_vespene < 150:
                return FAILED_COMMAND
        elif action == "research_TerranVehicleWeapons":
            if len(completed_armorys) <= 0 or have_research_TerranVehicleWeapons_level_3:
                return FAILED_COMMAND
            if have_research_TerranVehicleWeapons_level_2:
                if player_mineral < 250 or player_vespene < 250:
                    return FAILED_COMMAND
            if have_research_TerranVehicleWeapons_level_1:
                if player_mineral < 175 or player_vespene < 175:
                    return FAILED_COMMAND
            if player_mineral < 100 or player_vespene < 100:
                return FAILED_COMMAND
        elif action == "research_TerranShipWeapons":
            if len(completed_armorys) <= 0 or have_research_TerranShipWeapons_level_3:
                return FAILED_COMMAND
            if have_research_TerranShipWeapons_level_2:
                if player_mineral < 250 or player_vespene < 250:
                    return FAILED_COMMAND
            if have_research_TerranShipWeapons_level_1:
                if player_mineral < 175 or player_vespene < 175:
                    return FAILED_COMMAND
            if player_mineral < 100 or player_vespene < 100:
                return FAILED_COMMAND
        elif action == "research_TerranVehicleAndShipPlating":
            if len(completed_armorys) <= 0 or have_research_TerranVehicleAndShipPlating_level_3:
                return FAILED_COMMAND
            if have_research_TerranVehicleAndShipPlating_level_2:
                if player_mineral < 250 or player_vespene < 250:
                    return FAILED_COMMAND
            if have_research_TerranVehicleAndShipPlating_level_1:
                if player_mineral < 175 or player_vespene < 175:
                    return FAILED_COMMAND
            if player_mineral < 100 or player_vespene < 100:
                return FAILED_COMMAND
        return 0

    def get_state(self, obs):

        player_mineral = obs.observation.player.minerals
        player_vespene = obs.observation.player.vespene

        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        idle_scvs = [scv for scv in scvs if scv.order_length == 0]
        command_centers = self.get_my_units_by_type(
            obs, units.Terran.CommandCenter)

        refinerys = self.get_my_units_by_type(obs, units.Terran.Refinery)
        completed_refinerys = self.get_my_completed_units_by_type(
            obs, units.Terran.Refinery)
        notfull_completed_refinerys = self.get_notfull_worker_building_by_type(obs, units.Terran.Refinery)

        supply_depots = self.get_my_units_by_type(
            obs, units.Terran.SupplyDepot)
        completed_supply_depots = self.get_my_completed_units_by_type(
            obs, units.Terran.SupplyDepot)

        barrackses = self.get_my_units_by_type(obs, units.Terran.Barracks)
        completed_barrackses = self.get_my_completed_units_by_type(
            obs, units.Terran.Barracks)

        barrackstechlabs = self.get_my_units_by_type(obs, units.Terran.BarracksTechLab)
        completed_barrackstechlabs = self.get_my_completed_units_by_type(
            obs, units.Terran.BarracksTechLab)

        barracksreactors = self.get_my_units_by_type(obs, units.Terran.BarracksReactor)
        completed_barracksreactors = self.get_my_completed_units_by_type(
            obs, units.Terran.BarracksReactor)

        ghostacademys = self.get_my_units_by_type(obs, units.Terran.GhostAcademy)
        complete_ghostacademys = self.get_my_completed_units_by_type(obs, units.Terran.GhostAcademy)

        engineeringbays = self.get_my_units_by_type(
            obs, units.Terran.EngineeringBay)
        completed_engineeringbays = self.get_my_completed_units_by_type(
            obs, units.Terran.EngineeringBay)

        factorys = self.get_my_units_by_type(obs, units.Terran.Factory)
        completed_factorys = self.get_my_completed_units_by_type(obs, units.Terran.Factory)

        factorytechlabs = self.get_my_units_by_type(obs, units.Terran.FactoryTechLab)
        completed_factorytechlabs = self.get_my_completed_units_by_type(obs, units.Terran.FactoryTechLab)

        factoryreactors = self.get_my_units_by_type(obs, units.Terran.FactoryReactor)
        completed_factoryreactors = self.get_my_completed_units_by_type(obs, units.Terran.FactoryReactor)

        armorys = self.get_my_units_by_type(obs, units.Terran.Armory)
        completed_armorys = self.get_my_completed_units_by_type(obs, units.Terran.Armory)

        starports = self.get_my_units_by_type(obs, units.Terran.Starport)
        complete_starports = self.get_my_completed_units_by_type(obs, units.Terran.Starport)

        starportstechlabs = self.get_my_units_by_type(obs, units.Terran.StarportTechLab )
        completed_starportstechlabs = self.get_my_completed_units_by_type(obs, units.Terran.StarportTechLab )

        starportsreactors = self.get_my_units_by_type(obs, units.Terran.StarportReactor )
        completed_starportsreactors = self.get_my_completed_units_by_type(obs, units.Terran.StarportReactor)

        fusioncores = self.get_my_units_by_type(obs, units.Terran.FusionCore)
        complete_fusioncores = self.get_my_completed_units_by_type(obs, units.Terran.FusionCore)

        player_food_used = obs.observation.player.food_used
        player_food_cap = obs.observation.player.food_cap
        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)

        return (self.base_top_left,
                player_mineral,
                player_vespene,
                len(command_centers),
                len(scvs),
                len(idle_scvs),
                len(refinerys),
                len(completed_refinerys),
                len(notfull_completed_refinerys),
                len(supply_depots),
                len(completed_supply_depots),
                len(barrackses),
                len(completed_barrackses),
                len(barrackstechlabs),
                len(completed_barrackstechlabs),
                len(barracksreactors),
                len(completed_barracksreactors),
                len(ghostacademys),
                len(complete_ghostacademys),
                len(engineeringbays),
                len(completed_engineeringbays),
                len(factorys),
                len(completed_factorys),
                len(factorytechlabs),
                len(completed_factorytechlabs),
                len(factoryreactors),
                len(completed_factoryreactors),
                len(armorys),
                len(completed_armorys),
                len(starports),
                len(complete_starports),
                len(starportstechlabs),
                len(completed_starportstechlabs),
                len(starportsreactors),
                len(completed_starportsreactors),
                len(fusioncores),
                len(complete_fusioncores),
                free_supply,
                player_food_used,
                player_food_cap
                )

    def step(self, obs):
        super(SubAgent_Economic, self).step(obs)

        self.episode += 1
        state = self.get_state(obs)
        log.debug(f"state: {state}")
        action, action_idx = self.select_action(state)
        log.info(action)




        if self.previous_action is not None:
            step_reward = self.get_reward(obs)

            log.log(LOG_REWARD, "economic reward = " + str(obs.reward + step_reward))
            if not obs.last():
                self.memory.push(torch.Tensor(self.previous_state),
                                 F.one_hot(torch.LongTensor([self.previous_action_idx]), self.action_size)[0],
                                 torch.Tensor(state),
                                 torch.Tensor([obs.reward + step_reward]))

                self.optimize_model()
            else:
                pass
        else: # first step
            pass

        if self.episode % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.previous_state = state
        self.previous_action = action
        self.previous_action_idx = action_idx
        return getattr(self, action)(obs)

    def get_reward(self, obs):
        total_value_units_score = obs.observation.score_cumulative.total_value_units
        total_value_structures_score = obs.observation.score_cumulative.total_value_structures
        total_spent_minerals = obs.observation.score_cumulative.spent_minerals
        total_spent_vespene = obs.observation.score_cumulative.spent_vespene

        positive_reward = 0
        ## Positive reward update in this step

        ## Negative reward update in next step

        if total_spent_minerals > self.previous_total_spent_minerals:
            positive_reward += MORE_MINERALS_USED_REWARD_RATE * \
                (total_spent_minerals - self.previous_total_spent_minerals)
        if total_spent_vespene > self.previous_total_spent_vespene:
            positive_reward += MORE_VESPENE_USED_REWARD_RATE * \
                (total_spent_vespene - self.previous_total_spent_vespene)

        step_reward = positive_reward - self.negative_reward
        self.negative_reward = self.get_negative_reward(obs, self.previous_action)

        self.previous_total_value_units_score = total_value_units_score
        self.previous_total_value_structures_score = total_value_structures_score
        self.previous_total_spent_minerals = total_spent_minerals
        self.previous_total_spent_vespene = total_spent_vespene
        return step_reward

    def set_top_left(self, obs):
        if obs.first():
            command_center = self.get_my_units_by_type(
                obs, units.Terran.CommandCenter)[0]
            self.base_top_left = (command_center.x < 32)

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), dtype=torch.uint8, device=device)

        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(
            state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_net(
            non_final_next_states)

        expected_state_action_values = (
            next_state_values * GAMMA) + reward_batch

        loss = nn.CrossEntropyLoss(state_action_values,
                                expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

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
            log.log(LOG_MODEL, "Save memory economic")
