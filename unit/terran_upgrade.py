from unit.units import Upgrade

"""
class name from pysc2.lib.action
delete "research" and "quick"
"""

class TerranInfantryArmors(Upgrade):
    index = 11
    level = 0
    max_level = 3

    mineral_price = 100
    vespene_price = 100

    mp_gap = 75
    vp_gap = 75

    research_from = ["EngineeringBay"]
    requirements = []
    lv2_requirement = ["Armory"]

class TerranInfantryWeapons(Upgrade):
    index = 7
    level = 0
    max_level = 3

    mineral_price = 100
    vespene_price = 100

    mp_gap = 75
    vp_gap = 75

    research_from = ["EngineeringBay"]
    requirements = []
    lv2_requirement = ["Armory"]

class HiSecAutoTracking(Upgrade):
    index = 5
    level = 0
    max_level = 1

    mineral_price = 100
    vespene_price = 100

    research_from = ["EngineeringBay"]
    requirements = []

class TerranStructureArmor(Upgrade):
    index = 6
    level = 0
    max_level = 1

    mineral_price = 150
    vespene_price = 150

    research_from = ["EngineeringBay"]
    requirements = []

class CombatShield(Upgrade):
    index = 16
    level = 0
    max_level = 1

    mineral_price = 100
    vespene_price = 100

    research_from = ["BarracksTechLab"]
    requirements = []

class ConcussiveShells(Upgrade):
    index = 17
    level = 0
    max_level = 1

    mineral_price = 100
    vespene_price = 100

    research_from = ["BarracksTechLab"]
    requirements = []

class Stimpack(Upgrade):
    index = 15
    level = 0
    max_level = 1

    mineral_price = 50
    vespene_price = 50

    research_from = ["BarracksTechLab"]
    requirements = []

class InfernalPreigniter(Upgrade):
    index = 19
    level = 0
    max_level = 1

    mineral_price = 100
    vespene_price = 100

    research_from = ["FactoryTechLab"]
    requirements = []

class DrillingClaws(Upgrade):
    index = 122
    level = 0
    max_level = 1

    mineral_price = 75
    vespene_price = 75

    research_from = ["FactoryTechLab"]
    requirements = []

class CycloneLockOnDamage(Upgrade):
    # Mag-Field Accelerator
    index = 144
    level = 0
    max_level = 1

    mineral_price = 100
    vespene_price = 100

    research_from = ["FactoryTechLab"]
    requirements = ["Armory"]

class SmartServos(Upgrade):
    index = 289
    level = 0
    max_level = 1

    mineral_price = 100
    vespene_price = 100

    research_from = ["FactoryTechLab"]
    requirements = ["Armory"]

class CorvidReactor(Upgrade):
    index = 22
    level = 0
    max_level = 1

    mineral_price = 150
    vespene_price = 150

    research_from = ["StarportTechLab"]
    requirements = []

class CloakingField(Upgrade):
    index = 20
    level = 0
    max_level = 1

    mineral_price = 100
    vespene_price = 100

    research_from = ["StarportTechLab"]
    requirements = []

class HyperflightRotors(Upgrade):
    index = 136
    level = 0
    max_level = 1

    mineral_price = 150
    vespene_price = 150

    research_from = ["StarportTechLab"]
    requirements = []

class TerranVehicleWeapons(Upgrade):
    index = 30
    level = 0
    max_level = 3

    mineral_price = 100
    vespene_price = 100

    mp_gap = 75
    vp_gap = 75

    research_from = ["Armory"]
    requirements = []

class TerranShipWeapons(Upgrade):
    index = 36
    level = 0
    max_level = 3

    mineral_price = 100
    vespene_price = 100

    mp_gap = 75
    vp_gap = 75

    research_from = ["Armory"]
    requirements = []

class TerranVehicleAndShipArmors(Upgrade):
    index = 116
    level = 0
    max_level = 3

    mineral_price = 100
    vespene_price = 100

    mp_gap = 75
    vp_gap = 75

    research_from = ["Armory"]
    requirements = []
