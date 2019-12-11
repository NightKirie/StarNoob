import json
from event import *
import sc2reader
import os
import sys

sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 '../../lib'))


class PrefixSpan():
    def __init__(self, db):
        """
        Args:
            db (list[list[events]]): database

        """
        self._db = db

        # parameter

        # convert object to string
        self._str_db = []
        for replay in self._db:
            replay_str = []
            for event in replay:
                replay_str.append(str(event))
            self._str_db.append(replay_str)

        # 出現多少次，才會將 pattern 加入 prefix
        self.key = len(self._str_db) // 4

        # get prefix
        self._prefix = self.get_prefix()  # List[]

    def get_prefix(self):
        """ from db to parse prefix

        Returns:
            list: a list of single prefix

        """
        # 得到所有出現過的event作為pattern
        # 並且紀錄event出現在幾個replay中
        pattern = {}
        for replay in self._str_db:
            pattern_tmp = []  # to check whether count the pattern
            for event in replay:
                if event not in pattern.keys():
                    pattern[event] = 1
                    pattern_tmp.append(event)
                elif event not in pattern_tmp:
                    pattern[event] += 1
                    pattern_tmp.append(event)
                else:
                    continue

        # 刪除出現次數 < key 的 pattern
        result_pattern = []
        for prefix, key in pattern.items():
            if key > self.key:
                result_pattern.append(prefix)

        return result_pattern

    def mine(self, prefix):
        """ from prefix to mine postfix

        Args:
            prefix (list)

        Returns:
            list: postfix of input prefix
        """

        next_prefix = {}
        for i, replay in zip(range(len(self._str_db)), self._str_db):

            tmp_replay = list(replay)
            kmp_result = 0
            tmp_pattern = []  # to check whether count the pattern

            while tmp_replay != []:
                kmp_result = kmp(prefix, tmp_replay)
                if kmp_result != -1:

                    try:
                        # replay 中，prefix 下一個接的 pattern
                        new_match = tmp_replay[kmp_result+len(prefix)]
                    except:
                        break

                    if new_match not in next_prefix.keys():
                        next_prefix[new_match] = 1
                        tmp_pattern.append(new_match)
                    elif new_match not in tmp_pattern:
                        next_prefix[new_match] += 1
                        tmp_pattern.append(new_match)
                    else:
                        continue

                    tmp_replay = tmp_replay[kmp_result+1:]

                else:
                    break

        print(len(next_prefix))
        result_next_prefix = []
        for prefix, key in next_prefix.items():
            if key > self.key:
                result_next_prefix.append(prefix)

        return result_next_prefix

    def delete_prefix_in_db(self):
        """ 刪掉 self.db 中數量小於 key 的 prefix """
        for replay in self._str_db:
            for event in replay:
                if event not in self._prefix:
                    replay.pop(event)
