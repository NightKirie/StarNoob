#!/usr/bin/env python3
import sc2reader
import os
import sys

STARNOOB_LIB_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__FILE__)),
    '../../lib')

sys.path.append(STARNOOB_LIB_DIR)


def parse_replay(replay, player_no):
    result = []
    for event in replay.game_events:
        try:
            if event.player == replay.players[player_no-1]:
                result.append(event)
        except:
            continue

    return result


if __name__ == "__main__":
    path = ['./Acid-Plant-LE-13.sc2replay', 1]
    replay = sc2reader.load_replay(path[0], load_map=True)
    control_group_list = [0]*10
    selected_units = {}
    prev_str = ''

    for act in parse_replay(replay, path[1]):
        if type(act) == sc2reader.events.game.SelectionEvent:
            if act.new_units:
                selected_units = act.new_units
            else:
                selected_units = act.new_unit_info

        elif type(act) == sc2reader.events.game.GetControlGroupEvent:
            selected_units = control_group_list[act.control_group]

        elif type(act) == sc2reader.events.game.SetControlGroupEvent:
            control_group_list[act.control_group] = selected_units

        elif type(act) == sc2reader.events.game.CameraEvent:
            pass

        else:
            if selected_units:
                print('SelectionEvent', selected_units)
                selected_units = []

            now_str = str(act)
            if prev_str != now_str:
                print(now_str)
                prev_str = now_str
