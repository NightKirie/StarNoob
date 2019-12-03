import glob

# get all replays recursively
result = []
dir_path = r"./"
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

print(result)

