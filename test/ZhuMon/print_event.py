import os
import sys
# Import lib from parent dir
# Probably there's a better way of doing this.
sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 '../../'))

import lib.sc2reader as sc2reader

def parse_replay(replay):
    """
    type: 
        Dict{replay_dict} 
        v 
        List[List[sc2reader.events.game.GameEvent], ...] 
        # of inside List based on # of players
    """
    players = list(replay.player.values())
    result = [[] for p in players] # create List[List[], List[], ...] 
    for event in replay.game_events:
        try:
            player_of_event = players.index(event.player)
            result[player_of_event].append(event)
        except:
            continue

    return result

if __name__ == "__main__":
    replay = sc2reader.load_replay('./replay_for_test.SC2Replay', load_map=True)
    result = parse_replay(replay)
    for r in result:
        for e in r:
            print(e)

