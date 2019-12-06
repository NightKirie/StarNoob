#!/usr/bin/env python3

import os
import sys
import glob

STARNOOB_LIB_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__FILE__)),
                    '../../lib')

sys.path.append(STARNOOB_LIB_DIR)

import sc2reader

def get_all_replays_recursive(dir_path = r"./"):
    """ get all replays recursively
    
    Args: 
        dir_path (r"str"): root path to recursively find
        
    Returns:
        list["replay_path"]: a list of replay paths 
    """
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

def parse_replay(replay, player_no = 1):
    """ parse replay to game events

    Args:
        replay (sc2reader.resource.Replay): produced by sc2reader
        player_no (int): 1 or 2

    Returns:
        list[sc2reader.events.game.GameEvent, ...] : # of inside List based on # of players
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
    """ use sc2reader.load_replay and parse_replay
        to parse paths to game events
    
    Args:
        paths (list[list["path", player_no]]): paths of .SC2Replay
                        (from classify_by_race)
        or
        paths (list["path"]): paths of .SC2Replay
                        (from classify_by_map)
    
    Returns:
        list[list[events], ...]: a list of all events of replay in input paths
    """
    results = []
    if type(paths[0]) == list:
        for path in paths:
            replay = sc2reader.load_replay(path[0], load_map=True)
            results.append(parse_replay(replay, path[1]))
    elif type(paths[0]) == str:
        for path in paths:
            replay = sc2reader.load_replay(path, load_map=True)
            # suppose num of player is 2
            results.append(parse_replay(replay, 1))
            results.append(parse_replay(replay, 2))
    
    return results

def classify_by_map(paths, file_name = "classify_map.txt"):
    """ classify replays in paths by map

    Args:
        paths (list): a list of paths of replay file
        file_name (str): file path to write in classified path 
                        (default is classify_map.txt)

    Returns:
        Dict{"map_name":list["path"]}: a dictionary which key is map name and value is a list of replay path

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

    if file_name != None:
        with open(file_name, "w") as f:
            json.dump(maps, f)
    
    return maps

def classify_by_race(paths, file_name = "classify_race.txt"):
    """ classify replays in paths by races

    Args:
        paths (list): a list of path of replay filesi
        file_name (str): file path to write in classified path 
                        (default is classify_race.txt)

    Returns:
        dict{"Terran" :[["path", player_no],...],
             "Protoss":[["path", player_no],...],
             "Zerg"   :[["path", player_no],...]}
    """
    races = {"Terran": [], "Protoss": [], "Zerg": []}
    for path in paths:
        try:
            replay = sc2reader.load_replay(path, load_level=2)
            for i in range(len(replay.players)):
                races[replay.attributes[i+1]["Race"]].append([path, i+1])
        except:
            continue

    if file_name != None:
        with open(file_name, "w") as f:
            json.dump(races, f)

    return races

def get_replays_by_map(file_name = 'classify_map.txt', map_name= None):
    """ from file to get paths, and get replay path by map
        if map_name == None, return replay path which map used the most times
    
    Args: 
        file_name (str): name of file to load replay path
                        (default is 'classify_map.txt')
        map_name  (str): name of map to find in file
                        (default is None)
    
    Returns:
        str : map_name from input or map used the most time
        list: a list of paths
    """
    with open('classify_map.txt', 'r') as f:
        maps = json.load(f)
    
    if map_name is not None and map_name in maps.keys():
        return map_name, maps[map_name]
    else:
        max_num_map = ""
        max_num = 0
        for map_name, replays in maps.items():
            if len(replays) > max_num:
                max_num = len(replays)
                max_num_map = map_name
        return max_num_map, maps[max_num_map]

def get_replays_by_race(file_name = 'classify_race.txt', race = None):
    """ from file to get paths, and get replay path by race
        if race == None, return replay path which race used the most times
    
    Args: 
        file_name (str): name of file to load replay path
                        (default is 'classify_race.txt')
        race      (str): name of race to find in file
                        (default is None)
    
    Returns:
        str:  "Terran", "Protoss", "Zerg"
        list: a list of paths
    """
    with open(file_name, 'r') as f:
        races = json.load(f)
    
    if race not in ["Terran", "Protoss", "Zerg"]:
        return race, max(races.values())
    return race, races[race]

if __name__ == "__main__":
    paths = get_all_replays_recursive()
    # write into file
    maps = classify_by_map(paths)
    # get from file 
    map_name, replays_path = get_replays_by_map()
    
    # write into file
    races = classify_by_race(replays_path) # or maps[map_name]
    # get from file
    race_name, replays_path = get_replays_by_race()

    replays = load_replays(replays_path)

    # print all events of all replay (classified)
    for events in replays:
        for event in events:
            print(event)

    
