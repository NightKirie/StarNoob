import glob

# get all replays recursively
def get_all_replays_recursive(dir_path = r"./"):
    """ 
    type: 
        r"str" > List["replay_path"] 
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

if __name__ == "__main__":
    dir_path = r"./"
    get_all_replays_recursive(dir_path)
    print(result)

