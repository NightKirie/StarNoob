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


class CommandCenter(TerranBuilding):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self):
        self.mineral_price = 400
        self.gas_price = 0
        self.build_time = 71
        self.food_supply = 15

        self.hp = 1500
        self.attack = 0
        self.armor = 1
        self.range = 0

        self.requirements = []


class SupplyDepot(TerranBuilding):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self):
        self.mineral_price = 100
        self.gas_price = 0
        self.build_time = 21
        self.food_supply = 8

        self.hp = 400
        self.attack = 0
        self.armor = 1
        self.range = 0

        self.requirements = []


class Barracks(TerranBuilding):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self):
        self.mineral_price = 150
        self.gas_price = 0
        self.build_time = 46
        self.food_supply = 0

        self.hp = 1000
        self.attack = 0
        self.armor = 1
        self.range = 0
        self.movement = 1

        self.requirements = ["SupplyDepot"]


class OrbitalCommand(TerranBuilding):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self):
        self.mineral_price = 150
        self.gas_price = 0
        self.build_time = 25
        self.food_supply = 15

        self.hp = 1500
        self.attack = 0
        self.armor = 1
        self.range = 0

        self.requirements = ["CommandCenter", "Barracks"]


class PlanetaryFortress(TerranBuilding):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self):
        self.mineral_price = 150
        self.gas_price = 150
        self.build_time = 36
        self.food_supply = 15

        self.hp = 1500
        self.attack = 0
        self.armor = 3
        self.range = 6

        self.requirements = ["CommandCenter", "EngineeringBay"]

class Refinery(TerranBuilding):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self):
        self.mineral_price = 75
        self.gas_price = 0
        self.build_time = 21
        self.food_supply = 0

        self.hp = 500
        self.attack = 0
        self.armor = 1
        self.range = 0

        self.requirements = []

class EngineerBay(TerranBuilding):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self):
        self.mineral_price = 125
        self.gas_price = 0
        self.build_time = 25
        self.food_supply = 0

        self.hp = 850
        self.attack = 0
        self.armor = 1
        self.range = 0

        self.requirements = ["CommandCenter"]

class Bunker(TerranBuilding):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self):
        self.mineral_price = 100
        self.gas_price = 0
        self.build_time = 29

        self.hp = 400
        self.armor = 1

        self.requirements = ["Barracks"]


class MissileTurret(TerranBuilding):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self):
        self.mineral_price = 100
        self.gas_price = 0
        self.build_time = 18

        self.hp = 250
        self.armor = 0

        self.requirements = ["EngineeringBay"]


class SensorTower(TerranBuilding):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self):
        self.mineral_price = 125
        self.gas_price = 100
        self.build_time = 18

        self.hp = 200
        self.armor = 0

        self.requirements = ["EngineeringBay"]

class Factory(TerranBuilding):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self):
        self.mineral_price = 150
        self.gas_price = 100
        self.build_time = 43

        self.hp = 1250
        self.armor = 1

        self.requirements = ["Barracks"]

class GhostAcademy(TerranBuilding):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self):
        self.mineral_price = 150
        self.gas_price = 50
        self.build_time = 29

        self.hp = 1250
        self.armor = 1

        self.requirements = ["Barracks"]

class Armory(TerranBuilding):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self):
        self.mineral_price = 150
        self.gas_price = 100
        self.build_time = 46

        self.hp = 750
        self.armor = 1

        self.requirements = ["Factory"]

class Starport(TerranBuilding):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self):
        self.mineral_price = 150
        self.gas_price = 100
        self.build_time = 36

        self.hp = 1300
        self.armor = 1

        self.requirements = ["Factory"]

class FusionCore(TerranBuilding):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self):
        self.mineral_price = 150
        self.gas_price = 150
        self.build_time = 46

        self.hp = 750
        self.armor = 1

        self.requirements = ["Starport"]

class TechLab(TerranBuilding):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self):
        self.mineral_price = 50
        self.gas_price = 25
        self.build_time = 18

        self.hp = 400
        self.armor = 1

        self.requirements = []

class Reactor(TerranBuilding):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self):
        self.mineral_price = 50
        self.gas_price = 50
        self.build_time = 36

        self.hp = 400
        self.armor = 1

        self.requirements = []

class Marine(TerranCreature):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self):
        self.mineral_price = 50
        self.gas_price = 0
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

        self.build_from = ["Barracks"]
        self.requirements = []


class SCV(TerranCreature):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self):
        self.mineral_price = 50
        self.gas_price = 0
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

        self.build_from = ["CommandCenter"]
        self.requirements = []

class MULE(TerranCreature):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self):
        self.mineral_price = 0
        self.gas_price = 0
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

        self.build_from = []
        self.requirements = []

class Marauder(TerranCreature):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self):
        self.mineral_price = 100
        self.gas_price = 25
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

        self.build_from = ["Barracks"]
        self.requirements = ["TechLab"]

class Reaper(TerranCreature):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self):
        self.mineral_price = 50
        self.gas_price = 50
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

        self.build_from = ["Barracks"]
        self.requirements = []

class Ghost(TerranCreature):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self):
        self.mineral_price = 150
        self.gas_price = 125
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

        self.build_from = ["Barracks"]
        self.requirements = ["Ghost Academy", "TechLab"]

class Hellion(TerranCreature):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self):
        self.mineral_price = 100
        self.gas_price = 0
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

        self.build_from = ["Factory"]
        self.requirements = []

class Hellbat(TerranCreature):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self):
        self.mineral_price = 100
        self.gas_price = 0
        self.food_used = 2
        self.build_time = 21

        self.hp = 135
        self.armor = 0
        self.attribute = ['L', 'B', 'M']

        self.attack = 18
        self.range = 2
        self.dps = 12.6
        self.movement = 3.15

        self.build_from = ["Factory"]
        self.requirements = ["Armory"]

class WidowMine(TerranCreature):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self):
        self.mineral_price = 75
        self.gas_price = 25
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

        self.build_from = ["Factory"]
        self.requirements = []

class SiegeTanktm(TerranCreature):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self):
        self.mineral_price = 150
        self.gas_price = 125
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

        self.build_from = ["Factory"]
        self.requirements = ["Tech Lab"]

class SiegeTanksm(TerranCreature):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self):
        self.mineral_price = 150
        self.gas_price = 125
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

        self.build_from = ["Factory"]
        self.requirements = ["Tech Lab"]

class Cyclone(TerranCreature):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self):
        self.mineral_price = 150
        self.gas_price = 100
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

        self.build_from = ["Factory"]
        self.requirements = ["Tech Lab"]

class ThorExplosive(TerranCreature):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self):
        self.mineral_price = 300
        self.gas_price = 200
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

        self.build_from = ["Factory"]
        self.requirements = ["Tech Lab", "Armory"]


class ThorHighImpact(TerranCreature):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self):
        self.mineral_price = 300
        self.gas_price = 200
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

        self.build_from = ["Factory"]
        self.requirements = ["Tech Lab", "Armory"]

class Vikingfm(TerranCreature):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self):
        self.mineral_price = 150
        self.gas_price = 75
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

        self.build_from = ["Starport"]
        self.requirements = []

class Vikingam(TerranCreature):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self):
        self.mineral_price = 150
        self.gas_price = 75
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

        self.build_from = ["Starport"]
        self.requirements = []

class Medivac(TerranCreature):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self):
        self.mineral_price = 100
        self.gas_price = 100
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

        self.build_from = ["Starport"]
        self.requirements = []

class Liberatorfm(TerranCreature):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self):
        self.mineral_price = 150
        self.gas_price = 150
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

        self.build_from = ["Starport"]
        self.requirements = []

class Liberatordm(TerranCreature):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self):
        self.mineral_price = 150
        self.gas_price = 150
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

        self.build_from = ["Starport"]
        self.requirements = []

class Banshee(TerranCreature):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self):
        self.mineral_price = 150
        self.gas_price = 100
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

        self.build_from = ["Starport"]
        self.requirements = ["TechLab"]

class Raven(TerranCreature):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self):
        self.mineral_price = 100
        self.gas_price = 200
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

        self.build_from = ["Starport"]
        self.requirements = ["TechLab"]

class Battlecruiser(TerranCreature):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self):
        self.mineral_price = 400
        self.gas_price = 300
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

        self.build_from = ["Starport"]
        self.requirements = ["TechLab"]

class PlanetaryFortress(TerranCreature):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self):
        self.mineral_price = 150
        self.gas_price = 150
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

        self.build_from = []
        self.requirements = ["CommandCenter", "EngineeringBay"]

class MissileTurret(TerranCreature):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self):
        self.mineral_price = 100
        self.gas_price = 0
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

        self.build_from = []
        self.requirements = ["EngineeringBay"]

class AutoTurret(TerranCreature):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self):
        self.mineral_price = 0
        self.gas_price = 0
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

        self.build_from = []
        self.requirements = []

class PointDefenseDrone(TerranCreature):

    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self):
        self.mineral_price = 0
        self.gas_price = 0
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

        self.build_from = []
        self.requirements = []
