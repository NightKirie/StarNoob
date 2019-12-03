import os
import sys
# Import lib from parent dir
# Probably there's a better way of doing this.
sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 '../../'))

import lib.sc2reader as sc2reader

replay = sc2reader.load_replay('./replay_for_test.SC2Replay', load_map=True)

for i in replay.game_events:
    print(i)
