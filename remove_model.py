import os

REMOVE_LIST = ["agent", "battle", "economic", "training"]

if os.path.exists("./model"):
    os.chdir("./model")
    for file in REMOVE_LIST:
        if os.path.exists(file + "_dqn_policy"):
            os.remove(file + "_dqn_policy")
        if os.path.exists(file + "_dqn_target"):
            os.remove(file + "_dqn_target")
        if os.path.exists(file + "_memory"):
            os.remove(file + "_memory")