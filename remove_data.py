import os
if os.path.exists("AI_agent_data.gz"):
    os.remove("AI_agent_data.gz")
if os.path.exists("Sub_battle_data.gz"):
    os.remove("Sub_battle_data.gz")
if os.path.exists("Sub_building_data.gz"):
    os.remove("Sub_building_data.gz")
if os.path.exists("Sub_training_data.gz"):
    os.remove("Sub_training_data.gz")

import unit.terran_unit as terran

print(terran.SCV.hp)
print(terran.SCV().hp)