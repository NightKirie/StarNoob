from units import Upgrade

class TerranInfantryArmorsLevel1(Upgrade):
    def __init__(self):
        super().__init__()
        self.specialization()

    def specialization(self):
        self.name = self.__name__
        self.index = 11

        self.mineral_price = 100
        self.gas_price = 100
        self.build_time = 114

        self.research_from = ["EngineeringBay"]
        self.requirements = []
        self.affect = ["Ghost", "Marauder", "Reaper", "Marine"]
        
