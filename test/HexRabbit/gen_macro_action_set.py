#!/usr/bin/env python3
import os
import sys
import collections
import glob
import logging

STARNOOB_LIB_DIR = '/root/StarNoob/lib'
sys.path.append(STARNOOB_LIB_DIR)

import sc2reader

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)
LOG.addFilter(logging.Filter(__name__))

# To generate macro list, first do patching to game.py,
# then rm the __pycache__ dir in game.py's dir
'''
--- a/lib/sc2reader/events/game.py
+++ b/lib/sc2reader/events/game.py
@@ -251,13 +251,10 @@ class CommandEvent(GameEvent):
             string += "Right Click"

         if self.ability_type == "TargetUnit":
-            string += "; Target: {0} [{1:0>8X}]".format(
-                self.target.name, self.target_unit_id
+            string += "; Target: {0}".format(
+                self.target.name
             )

-        if self.ability_type in ("TargetPoint", "TargetUnit"):
-            string += "; Location: (%.2f, %.2f, %.2f)" % self.location
-
         return string
'''


# get all replays recursively
def get_all_replays_recursive(dir_path = r"./"):
    result = []
    replay_path = r"*.SC2Replay"
    all_dir_path = r"*/"
    tmp_replay = glob.glob(dir_path+replay_path)
    tmp_folder = glob.glob(dir_path+all_dir_path)
    result += tmp_replay

    while tmp_replay != [] or tmp_folder != []:
        dir_path += all_dir_path
        tmp_replay = glob.glob(dir_path+replay_path)
        tmp_folder = glob.glob(dir_path+all_dir_path)
        result += tmp_replay

    return result


def load_replays(paths):
    results = []
    total = len(paths)
    LOG.info(f'Total replays: {total}')
    for i, path in enumerate(paths):
        LOG.info(f'{i}/{total} Processing {path[0] if len(path[0]) < 50 else f"...{path[0][-50:]}"}')
        replay = sc2reader.load_replay(path[0], load_map=True)
        results.append(parse_replay(replay, path[1]))

    return results

def parse_replay(replay, player_no):
    result = []
    for event in replay.game_events:
        try:
            if event.player == replay.players[player_no-1]:
                result.append(event)
        except:
            continue

    return result

def classify_by_race(paths):
    """
    type:
        List["path", ...]
        v
        Dict{"Terran":[["path", player_no],], "Protoss":[["path", player_no]...], "Zerg":[["path",player_no],...]}
    """
    races = {"Terran": [], "Protoss": [], "Zerg": []}
    for path in paths:
        try:
            replay = sc2reader.load_replay(path, load_level=2)
            for i in range(len(replay.players)):
                races[replay.attributes[i+1]["Race"]].append([path, i+1])
        except:
            continue

    return races

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ['Terran', 'Protoss', 'Zerg']:
        print(f'Usage: {__file__} [Terran|Protoss|Zerg]')
        sys.exit()

    lst = []
    mapping = {}
    cnt = collections.defaultdict(int)

    targets = get_all_replays_recursive(r'/root/replays/')
    targets = classify_by_race(targets)[sys.argv[1]]
    event_list_list = load_replays(targets)


    for event_list in event_list_list:
        control_group_list = [0]*10
        selected_units = []
        prev_str = ''
        for act in event_list:
            if type(act) == sc2reader.events.game.SelectionEvent:
                if act.new_units:
                    selected_units = act.new_units
                else:
                    selected_units = act.new_unit_info

            elif type(act) == sc2reader.events.game.GetControlGroupEvent:
                selected_units = control_group_list[act.control_group]

            elif type(act) == sc2reader.events.game.AddToControlGroupEvent:
                control_group_list[act.control_group] += selected_units

            elif type(act) == sc2reader.events.game.SetControlGroupEvent:
                control_group_list[act.control_group] = selected_units

            elif type(act) == sc2reader.events.game.CameraEvent:
                pass

            else:
                if selected_units:
                    lst.append('SelectionEvent ' + str(set([str(s).split(' ')[0] for s in selected_units])))
                    selected_units = []

                now_str = str(act)
                if prev_str != now_str:
                    lst.append(now_str)
                    prev_str = now_str


    for l in [3, 4, 5, 6]:
        for i in range(len(lst) - l):
            if lst[i].startswith('Sele'): # SelectionEvent
                conlst = tuple(lst[i:i+l])
                val = hash(conlst)
                mapping[val] = conlst
                cnt[val] += 1

    i = 0
    for k in sorted(cnt, key=cnt.get, reverse=True):
        if i > 500:
            break
        print(len(mapping[k]), cnt[k], mapping[k])
        i += 1

if __name__ == '__main__':
    main()

