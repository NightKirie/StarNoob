from unit.units import Building
from unit.units import Creature


class TerranBuilding(Building):

    def __init__(self):
        super().__init__()

    def getEquivalentHP(self, attack):
        if attack == 0:
            return self.hp
        else:
            return self.hp * attack / max(attack - self.armor, 1)


class TerranCreature(Creature):

    def __init__(self):
        super().__init__()

    def getEquivalentHP(self, attack):
        if attack == 0:
            return self.hp
        else:
            return self.hp * attack / max(attack - self.armor, 1)


class TechLab(TerranBuilding):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self): 
        self.name = "TechLab"
        self.mineral_price = 50
        self.vespene_price = 25
        self.build_time = 18

        self.hp = 400
        self.armor = 1
        self.index = 5
        self.requirements = []


class Reactor(TerranBuilding):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self): 
        self.name = "Reactor"
        self.mineral_price = 50
        self.vespene_price = 50
        self.build_time = 36

        self.hp = 400
        self.armor = 1
        self.index = 6
        self.requirements = []


class CommandCenter(TerranBuilding):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self): 
        self.name = "CommandCenter"
        self.mineral_price = 400
        self.vespene_price = 0
        self.build_time = 71
        self.food_supply = 15

        self.hp = 1500
        self.attack = 0
        self.armor = 1
        self.range = 0
        self.index = 18
        self.requirements = []


class SupplyDepot(TerranBuilding):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self): 
        self.name = "SupplyDepot"      
        self.mineral_price = 100
        self.vespene_price = 0
        self.build_time = 21
        self.food_supply = 8

        self.hp = 400
        self.attack = 0
        self.armor = 1
        self.range = 0
        self.index = 19
        self.requirements = []


class Barracks(TerranBuilding):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self): 
        self.name = "Barracks"
        self.mineral_price = 150
        self.vespene_price = 0
        self.build_time = 46
        self.food_supply = 0

        self.hp = 1000
        self.attack = 0
        self.armor = 1
        self.range = 0
        self.movement = 1
        self.index = 21
        self.requirements = ["SupplyDepot"]

class BarracksTechLab(TechLab):
    def specialization(self): 
        self.name = self.__name__
        self.index = 37
        self.requirements = ["Barracks"]

class BarracksReactor(Reactor):
    def specialization(self): 
        self.name = self.__name__
        self.index = 38
        self.requirements = ["Barracks"]

class OrbitalCommand(TerranBuilding):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self): 
        self.name = "OrbitalCommand"
        self.mineral_price = 150
        self.vespene_price = 0
        self.build_time = 25
        self.food_supply = 15

        self.hp = 1500
        self.attack = 0
        self.armor = 1
        self.range = 0
        self.index = 132
        self.requirements = ["CommandCenter", "Barracks"]


class PlanetaryFortress(TerranBuilding):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self): 
        self.name = "PlanetaryFortress"
        self.mineral_price = 150
        self.vespene_price = 150
        self.build_time = 36
        self.food_supply = 15

        self.hp = 1500
        self.attack = 0
        self.armor = 3
        self.range = 6
        self.index = 130
        self.requirements = ["CommandCenter", "EngineeringBay"]

class Refinery(TerranBuilding):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self): 
        self.name = "Refinery"
        self.mineral_price = 75
        self.vespene_price = 0
        self.build_time = 21
        self.food_supply = 0

        self.hp = 500
        self.attack = 0
        self.armor = 1
        self.range = 0
        self.index = 20
        self.requirements = []

class EngineeringBay(TerranBuilding):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self): 
        self.name = "EngineerBay"
        self.mineral_price = 125
        self.vespene_price = 0
        self.build_time = 25
        self.food_supply = 0

        self.hp = 850
        self.attack = 0
        self.armor = 1
        self.range = 0
        self.index = 22
        self.requirements = ["CommandCenter"]

class Bunker(TerranBuilding):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self): 
        self.name = "Bunker"
        self.mineral_price = 100
        self.vespene_price = 0
        self.build_time = 29

        self.hp = 400
        self.armor = 1

        self.requirements = ["Barracks"]


class MissileTurret(TerranBuilding):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self): 
        self.name = "MissileTurret"
        self.mineral_price = 100
        self.vespene_price = 0
        self.build_time = 18

        self.hp = 250
        self.armor = 0
        self.index = 24
        self.requirements = ["EngineeringBay"]


class SensorTower(TerranBuilding):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self): 
        self.name = "SensorTower"
        self.mineral_price = 125
        self.vespene_price = 100
        self.build_time = 18

        self.hp = 200
        self.armor = 0
        self.index = 25
        self.requirements = ["EngineeringBay"]

class Factory(TerranBuilding):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self): 
        self.name = "Factory"
        self.mineral_price = 150
        self.vespene_price = 100
        self.build_time = 43

        self.hp = 1250
        self.armor = 1
        self.index = 27
        self.requirements = ["Barracks"]

class FactoryTechLab(TechLab):
    def specialization(self):
        self.name = self.__name__
        self.index = 39
        self.requirements = ["Factory"]

class FactoryReactor(Reactor):
    def specialization(self):
        self.name = self.__name__
        self.index = 40
        self.requirements = ["Factory"]

class GhostAcademy(TerranBuilding):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self): 
        self.name = "GhostAcademy"
        self.mineral_price = 150
        self.vespene_price = 50
        self.build_time = 29

        self.hp = 1250
        self.armor = 1
        self.index = 26
        self.requirements = ["Barracks"]

class Armory(TerranBuilding):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self): 
        self.name = "Armory"
        self.mineral_price = 150
        self.vespene_price = 100
        self.build_time = 46

        self.hp = 750
        self.armor = 1
        self.index = 29
        self.requirements = ["Factory"]

class Starport(TerranBuilding):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self): 
        self.name = "Starport"
        self.mineral_price = 150
        self.vespene_price = 100
        self.build_time = 36

        self.hp = 1300
        self.armor = 1
        self.index = 28
        self.requirements = ["Factory"]

class StarportTechLab(TechLab):
    def specialization(self):
        self.name = self.__name__
        self.index = 41
        self.requirements = ["Starport"]

class StarportReactor(Reactor):
    def specialization(self):
        self.name = self .__name__
        self.index = 42
        self.requirements = ["Starport"]


class FusionCore(TerranBuilding):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self): 
        self.name = "FusionCore"
        self.mineral_price = 150
        self.vespene_price = 150
        self.build_time = 46

        self.hp = 750
        self.armor = 1
        self.index = 30
        self.requirements = ["Starport"]


class Marine(TerranCreature):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self): 
        self.name = "Marine"
        self.mineral_price = 50
        self.vespene_price = 0
        self.food_used = 1
        self.build_time = 18

        self.hp = 45
        self.armor = 0
        self.attribute = ['L', 'B']

        self.attack = 6
        self.range = 5
        self.dps = 9.8
        self.bonus_attack = {}
        self.movement = 3.15
        self.index = 48
        self.build_from = ["Barracks"]
        self.requirements = []


class SCV(TerranCreature):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self): 
        self.name = "SCV"
        self.mineral_price = 50
        self.vespene_price = 0
        self.food_used = 1
        self.build_time = 12

        self.hp = 45
        self.armor = 0
        self.attribute = ['L', 'B', 'M']

        self.attack = 5
        self.range = 0
        self.dps = 4.7
        self.bonus_attack = {}
        self.movement = 3.94
        self.index = 45
        self.build_from = ["CommandCenter"]
        self.requirements = []

class MULE(TerranCreature):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self): 
        self.name = "MULE"
        self.mineral_price = 0
        self.vespene_price = 0
        self.food_used = 0
        self.build_time = 0

        self.hp = 60
        self.armor = 0
        self.attribute = ['L', 'M']

        self.attack = 0
        self.range = 0
        self.dps = 0
        self.bonus_attack = {}
        self.movement = 3.94
        self.index = 268
        self.build_from = []
        self.requirements = []

class Marauder(TerranCreature):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self): 
        self.name = "Marauder"
        self.mineral_price = 100
        self.vespene_price = 25
        self.food_used = 2
        self.build_time = 21

        self.hp = 125
        self.armor = 1
        self.attribute = ['A', 'B']

        self.attack = 5
        self.range = 6
        self.dps = 9.3
        self.bonus_attack = {'A': 5}
        self.movement = 3.15
        self.index = 51
        self.build_from = ["Barracks"]
        self.requirements = ["TechLab"]

class Reaper(TerranCreature):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self): 
        self.name = "Reaper"
        self.mineral_price = 50
        self.vespene_price = 50
        self.food_used = 1
        self.build_time = 32

        self.hp = 60
        self.armor = 0
        self.attribute = ['L', 'B']

        self.attack = 4
        self.range = 5
        self.dps = 10.1
        self.bonus_attack = {}
        self.movement = 5.25
        self.index = 49
        self.build_from = ["Barracks"]
        self.requirements = []

class Ghost(TerranCreature):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self): 
        self.name = "Ghost"
        self.mineral_price = 150
        self.vespene_price = 125
        self.food_used = 2
        self.build_time = 29

        self.hp = 100
        self.armor = 0
        self.attribute = ['B', 'P']

        self.attack = 10
        self.range = 6
        self.dps = 9.3
        self.bonus_attack = {'L': 10}
        self.movement = 3.94
        self.index = 50
        self.build_from = ["Barracks"]
        self.requirements = ["GhostAcademy", "TechLab"]

class Hellion(TerranCreature):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self): 
        self.name = "Hellion"
        self.mineral_price = 100
        self.vespene_price = 0
        self.food_used = 2
        self.build_time = 21

        self.hp = 90
        self.armor = 0
        self.attribute = ['L', 'M']

        self.attack = 8
        self.range = 5
        self.dps = 4.5
        self.bonus_attack = {'L': 6}
        self.movement = 5.95
        self.index = 53
        self.build_from = ["Factory"]
        self.requirements = []

class Hellbat(TerranCreature):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self): 
        self.name = "Hellbat"
        self.mineral_price = 100
        self.vespene_price = 0
        self.food_used = 2
        self.build_time = 21

        self.hp = 135
        self.armor = 0
        self.attribute = ['L', 'B', 'M']

        self.attack = 18
        self.range = 2
        self.dps = 12.6
        self.movement = 3.15
        self.index = 484
        self.build_from = ["Factory"]
        self.requirements = ["Armory"]

class WidowMine(TerranCreature):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self): 
        self.name = "WidowMine"
        self.mineral_price = 75
        self.vespene_price = 25
        self.food_used = 2
        self.build_time = 21

        self.hp = 90
        self.armor = 0
        self.attribute = ['L', 'M']

        self.attack = 125
        self.range = 5
        self.dps = 9.3
        self.bonus_attack = {'Shield': 35}
        self.movement = 3.94
        self.index = 498
        self.build_from = ["Factory"]
        self.requirements = []

class SiegeTank(TerranCreature):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self): 
        self.name = "SiegeTank"
        self.mineral_price = 150
        self.vespene_price = 125
        self.food_used = 3
        self.build_time = 32

        self.hp = 175
        self.armor = 1
        self.attribute = ['A', 'M']

        self.attack = 15
        self.range = 7
        self.dps = 20.3
        self.bonus_attack = {'A': 10}
        self.movement = 3.15
        self.index = 33
        self.build_from = ["Factory"]
        self.requirements = ["TechLab"]

class SiegeTankSieged(TerranCreature):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self): 
        self.name = "SiegeTankSieged"
        self.mineral_price = 150
        self.vespene_price = 125
        self.food_used = 3
        self.build_time = 32

        self.hp = 175
        self.armor = 1
        self.attribute = ['A', 'M']

        self.attack = 40
        self.range = 13
        self.dps = 18.69
        self.bonus_attack = {'A': 30}
        self.movement = 0
        self.index = 32
        self.build_from = ["Factory"]
        self.requirements = ["TechLab"]

class Cyclone(TerranCreature):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self): 
        self.name = "Cyclone"
        self.mineral_price = 150
        self.vespene_price = 100
        self.food_used = 3
        self.build_time = 32

        self.hp = 180
        self.armor = 1
        self.attribute = ['A', 'M']

        self.attack = 3
        self.range = 4
        self.dps = 30
        self.bonus_attack = {'A': 2}
        self.movement = 4.13
        self.index = 692
        self.build_from = ["Factory"]
        self.requirements = ["TechLab"]

class Thor(TerranCreature):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self): 
        self.name = "Thor"
        self.mineral_price = 300
        self.vespene_price = 200
        self.food_used = 6
        self.build_time = 43

        self.hp = 400
        self.armor = 1
        self.attribute = ['A', 'M', 'Ma']

        self.attack = 30
        self.range = 7
        self.dps = 33
        self.bonus_attack = {'L Air': 6}
        self.movement = 2.62
        self.index = 52
        self.build_from = ["Factory"]
        self.requirements = ["TechLab", "Armory"]


class ThorHighImpactMode(TerranCreature):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self): 
        self.name = "ThorHighImpactMode"
        self.mineral_price = 300
        self.vespene_price = 200
        self.food_used = 6
        self.build_time = 43

        self.hp = 400
        self.armor = 1
        self.attribute = ['A', 'M', 'Ma']

        self.attack = 30
        self.range = 7
        self.dps = 33
        self.bonus_attack = {'A Air': 15}
        self.movement = 2.62
        self.index = 691
        self.build_from = ["Factory"]
        self.requirements = ["TechLab", "Armory"]

class VikingFighter(TerranCreature):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self): 
        self.name = "VikingFighter"
        self.mineral_price = 150
        self.vespene_price = 75
        self.food_used = 2
        self.build_time = 30

        self.hp = 125
        self.armor = 0
        self.attribute = ['A', 'M']

        self.attack = 10
        self.range = 9
        self.dps = 14
        self.bonus_attack = {'A': 4}
        self.movement = 3.85
        self.index = 35
        self.build_from = ["Starport"]
        self.requirements = []

class Viking(TerranCreature):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self): 
        self.name = "Viking"
        self.mineral_price = 150
        self.vespene_price = 75
        self.food_used = 2
        self.build_time = 30

        self.hp = 125
        self.armor = 0
        self.attribute = ['A', 'M']

        self.attack = 12
        self.range = 6
        self.dps = 16.8
        self.bonus_attack = {'M': 8}
        self.movement = 3.15
        self.index = 34
        self.build_from = ["Starport"]
        self.requirements = []

class Medivac(TerranCreature):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self): 
        self.name = "Medivac"
        self.mineral_price = 100
        self.vespene_price = 100
        self.food_used = 2
        self.build_time = 30

        self.hp = 150
        self.armor = 1
        self.attribute = ['A', 'M']

        self.attack = 0
        self.range = 0
        self.dps = 0
        self.bonus_attack = {}
        self.movement = 3.5
        self.index = 54
        self.build_from = ["Starport"]
        self.requirements = []

class Liberator(TerranCreature):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self): 
        self.name = "Liberator"
        self.mineral_price = 150
        self.vespene_price = 150
        self.food_used = 3
        self.build_time = 43

        self.hp = 180
        self.armor = 0
        self.attribute = ['A', 'M']

        self.attack = 7
        self.range = 5
        self.dps = 10.9
        self.bonus_attack = {}
        self.movement = 4.72
        self.index = 689
        self.build_from = ["Starport"]
        self.requirements = []

class LiberatorAG(TerranCreature):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self): 
        self.name = "LiberatorAG"
        self.mineral_price = 150
        self.vespene_price = 150
        self.food_used = 3
        self.build_time = 43

        self.hp = 180
        self.armor = 0
        self.attribute = ['A', 'M']

        self.attack = 75
        self.range = 10
        self.dps = 65.8
        self.bonus_attack = {}
        self.movement = 0
        self.index = 734
        self.build_from = ["Starport"]
        self.requirements = []

class Banshee(TerranCreature):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self): 
        self.name = "Banshee"
        self.mineral_price = 150
        self.vespene_price = 100
        self.food_used = 3
        self.build_time = 43

        self.hp = 140
        self.armor = 0
        self.attribute = ['L', 'M']

        self.attack = 12
        self.range = 6
        self.dps = 27
        self.bonus_attack = {}
        self.movement = 3.85
        self.index = 55
        self.build_from = ["Starport"]
        self.requirements = ["TechLab"]

class Raven(TerranCreature):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self): 
        self.name = "Raven"
        self.mineral_price = 100
        self.vespene_price = 200
        self.food_used = 2
        self.build_time = 43

        self.hp = 140
        self.armor = 1
        self.attribute = ['L', 'M']

        self.attack = 0
        self.range = 0
        self.dps = 0
        self.bonus_attack = {}
        self.movement = 3.85
        self.index = 56
        self.build_from = ["Starport"]
        self.requirements = ["TechLab"]

class Battlecruiser(TerranCreature):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self): 
        self.name = "Battlecruiser"
        self.mineral_price = 400
        self.vespene_price = 300
        self.food_used = 6
        self.build_time = 64

        self.hp = 550
        self.armor = 3
        self.attribute = ['A', 'M', 'Ma']

        self.attack = 8
        self.range = 6
        self.dps = 50
        self.bonus_attack = {}
        self.movement = 2.62
        self.index = 57
        self.build_from = ["Starport"]
        self.requirements = ["TechLab"]

class PlanetaryFortress(TerranCreature):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self): 
        self.name = "PlanetaryFortress"
        self.mineral_price = 150
        self.vespene_price = 150
        self.food_used = 0
        self.build_time = 36

        self.hp = 1500
        self.armor = 3
        self.attribute = ['A', 'M', 'S']

        self.attack = 40
        self.range = 6
        self.dps = 28
        self.bonus_attack = {}
        self.movement = 0
        self.index = 130
        self.build_from = []
        self.requirements = ["CommandCenter", "EngineeringBay"]

class MissileTurret(TerranCreature):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self): 
        self.name = "MissileTurret"
        self.mineral_price = 100
        self.vespene_price = 0
        self.food_used = 0
        self.build_time = 18

        self.hp = 250
        self.armor = 0
        self.attribute = ['A', 'M', 'S']

        self.attack = 12
        self.range = 7
        self.dps = 42.1
        self.bonus_attack = {}
        self.movement = 0
        self.index = 23
        self.build_from = []
        self.requirements = ["EngineeringBay"]

class AutoTurret(TerranCreature):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self): 
        self.name = "AutoTurret"
        self.mineral_price = 0
        self.vespene_price = 0
        self.food_used = 0
        self.build_time = 0

        self.hp = 125
        self.armor = 1
        self.attribute = ['A', 'M', 'S']

        self.attack = 18
        self.range = 6
        self.dps = 31.58
        self.bonus_attack = {}
        self.movement = 0
        self.index = 31
        self.build_from = []
        self.requirements = []

class PointDefenseDrone(TerranCreature):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self): 
        self.name = "PointDefenseDrone"
        self.mineral_price = 0
        self.vespene_price = 0
        self.food_used = 0
        self.build_time = 0

        self.hp = 50
        self.armor = 0
        self.attribute = ['L', 'M', 'S']

        self.attack = 0
        self.range = 8
        self.dps = 0
        self.bonus_attack = {}
        self.movement = 0
        self.index = 11
        self.build_from = []
        self.requirements = []
