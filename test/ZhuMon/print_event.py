import os
import sys
# Import lib from parent dir
# Probably there's a better way of doing this.
sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 '../../'))

import lib.sc2reader as sc2reader

def parse_replay(replay, player_no):
    """
    type: 
        Dict{replay_dict} 
        v 
        List[sc2reader.events.game.GameEvent, ...] 
        # of inside List based on # of players
    """
    result = []
    for event in replay.game_events:
        try:
            if event.player == replay.players[player_no-1]:
                result.append(event)
        except:
            continue

    return result

def load_replays(paths):
    """
    type:
        List["string", "string",...]
        v
        List[List[replay_event, ...], ...]
    """
    results = []
    for path in paths:
        replay = sc2reader.load_replay(path[0], load_map=True)
        results.append(parse_replay(replay, path[1]))

    return results
        
def classify_by_map(paths):
    """
    type:
        List["path", ...]
        v
        Dict{"map_name":List["path",...], ...}
    """
    maps = {}
    for path in paths:
        try:
            replay = sc2reader.load_replay(path, load_level=1, load_map = True)
            if replay.map.name not in maps.keys():
                maps[replay.map.name] = [path]
            else:
                maps[replay.map.name].append(path)
        except:
            continue
    return maps

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


if __name__ == "__main__":
    replays_path = ['./replay_for_test.SC2Replay']
    print(classify_by_map(replays_path))
    # results = load_replays(replays_path)
    # for result in results:
    #    # for r in result:
    #        # for e in r:
    #             print(e)
