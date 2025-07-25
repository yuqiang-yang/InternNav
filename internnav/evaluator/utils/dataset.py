import os
import sys

import lmdb
import msgpack_numpy

from internnav import PROJECT_ROOT_PATH
from internnav.evaluator.utils.common import load_data
from internnav.evaluator.utils.config import get_lmdb_path, get_lmdb_prefix

from .config import Config


def split_data(config):
    if isinstance(config, dict):
        config = Config(**config)
    run_type = config.run_type
    split_number = 1  # config.total_rank
    run_type = run_type
    name = config.task_name
    split_data_types = config.split_data_types
    base_data_dir = config.base_data_dir
    filter_stairs = config.filter_stairs

    print(f'run_type:{run_type}')
    print(f'name:{name}')
    print(f'split_data_types:{split_data_types}')
    prefix = get_lmdb_prefix(run_type)
    if run_type == 'eval':
        filter_same_trajectory = False
    elif run_type == 'sample':
        filter_same_trajectory = True
    else:
        print(f'unknown run_type:{run_type}')
        sys.exit()

    lmdb_path = get_lmdb_path(name)
    # get all data
    path_key_map = {}
    count = 0
    for split_data_type in split_data_types:
        data_map = load_data(
            base_data_dir,
            split_data_type,
            filter_same_trajectory=filter_same_trajectory,
            filter_stairs=filter_stairs,
        )
        for scan, path_list in data_map.items():
            path_key_list = []
            for path in path_list:
                trajectory_id = path['trajectory_id']
                episode_id = path['episode_id']
                path_key = f'{trajectory_id}_{episode_id}'
                path_key_list.append(path_key)
            path_key_map[scan] = path_key_list
            count += len(path_key_list)

    print(f'TOTAL:{count}')

    # split rank
    rank_map = {}
    split_length = count // split_number
    index = -1
    for scan, path_key_list in path_key_map.items():
        for path_key in path_key_list:
            index += 1
            rank = index // split_length
            if rank >= split_number:
                rank = split_number - 1
            rank_map[path_key] = rank

    ranked_data = {}
    for i in range(split_number):
        filtered_path_key_map = {}
        for scan, path_key_list in path_key_map.items():
            filtered_list = []
            for path_key in path_key_list:
                if rank_map[path_key] == i:
                    filtered_list.append(path_key)
            if len(filtered_list) > 0:
                filtered_path_key_map[scan] = filtered_list
        ranked_data[i] = filtered_path_key_map

    for rank, path_key_map in ranked_data.items():
        count = 0
        for scan, path_key_list in path_key_map.items():
            count += len(path_key_list)
            print(f'[rank:{rank}][scan:{scan}][count:{len(path_key_list)}]')
        print(f'[rank:{rank}][count:{count}]')

    if not os.path.exists(lmdb_path):
        os.makedirs(lmdb_path)
    database = lmdb.open(
        f'{lmdb_path}/sample_data.lmdb',
        map_size=1 * 1024 * 1024 * 1024 * 1024,
        max_dbs=0,
    )
    with database.begin(write=True) as txn:
        for rank, path_key_map in ranked_data.items():
            key = f'{prefix}_{rank}'.encode()
            value = msgpack_numpy.packb(path_key_map, use_bin_type=True)
            txn.put(key, value)
            print(f'finish [key:{key}]')
    database.close()


class ResultLogger:
    def __init__(self, config):
        if isinstance(config, dict):
            config = Config(**config)
        self.name = config.task_name
        self.lmdb_path = get_lmdb_path(self.name)
        self.split_map = self.get_split_map(
            base_data_dir=config.base_data_dir,
            split_data_types=config.split_data_types,
            filter_stairs=config.filter_stairs
        )

    def get_split_map(
        self,
        base_data_dir,
        split_data_types,
        filter_stairs,
    ):
        split_map = {}
        for split_data_type in split_data_types:
            load_data_map = load_data(
                base_data_dir,
                split_data_type,
                filter_same_trajectory=False,
                filter_stairs=filter_stairs,
            )
            path_key_list = []
            for scan, path_list in load_data_map.items():
                for path in path_list:
                    trajectory_id = path['trajectory_id']
                    episode_id = path['episode_id']
                    path_key = f'{trajectory_id}_{episode_id}'
                    path_key_list.append(path_key)
            split_map[split_data_type] = path_key_list
        return split_map

    def write_now_result(self):

        # create log file
        log_content = []

        def log_print(content):
            log_content.append(str(content))

        self.database_read = lmdb.open(
            f'{self.lmdb_path}/sample_data.lmdb',
            map_size=1 * 1024 * 1024 * 1024 * 1024,
            readonly=True,
            lock=False,
        )

        for split, path_key_list in self.split_map.items():
            data_list = []
            for path_key in path_key_list:
                data_key = path_key
                with self.database_read.begin() as txn:
                    value = txn.get(data_key.encode())
                    if value is None:
                        continue
                    value = msgpack_numpy.unpackb(value)
                value['path_key'] = path_key
                data_list.append(value)
            count = len(data_list)
            log_print(f'[split:{split}] total {count} data')
            total_TL = 0
            total_NE = 0
            total_osr = 0
            total_success = 0
            total_spl = 0
            reason_map = {'reach_goal': 0}

            for data in data_list:
                # TL Trajectory Length (TL) - trajectory total length (0)
                TL = data['info']['TL']
                # NE Navigation Error (NE) - Euclidean distance from current position to goal (-1)
                NE = data['info']['NE']
                if NE < 0:
                    NE = 0
                # OS Oracle Success Rate (OSR) - whether there is a point in the trajectory reaching the goal (-1)
                osr = data['info']['osr']
                if osr < 0:
                    osr = 0
                # SR Success Rate (SR) - whether the goal is reached (0)
                success = data['info']['success']
                # SPL (Success weighted by Path Length)(0)
                spl = data['info']['spl']

                total_TL += TL
                total_NE += NE
                total_osr += osr
                total_success += success
                total_spl += spl

                ret_type = data['fail_reason']
                if ret_type == '':
                    ret_type = 'success'
                if ret_type not in reason_map:
                    reason_map[ret_type] = 1
                else:
                    reason_map[ret_type] = reason_map[ret_type] + 1
                if success > 0:
                    reason_map['reach_goal'] = reason_map['reach_goal'] + 1

            log_print(f'############[{split}]#############')
            if count == 0:
                log_print('############[count == 0,skip]#############')
                continue
            log_print(f'TL = {total_TL} / {count} = {round((total_TL / count),4)}')
            log_print(f'NE = {total_NE} / {count} = {round((total_NE / count),4)}')
            if 'fall' not in reason_map:
                reason_map['fall'] = 0
            log_print(f"FR = {reason_map['fall']} / {count} = {round((reason_map['fall'] / count),4) * 100}%")
            if 'stuck' in reason_map:
                log_print(f"StR = {reason_map['stuck']} / {count} = {round((reason_map['stuck'] / count),4) * 100}%")
            else:
                log_print(f'StR = 0 / {count} = 0%')
            log_print(f'OS = {total_osr} / {count} = {round((total_osr / count),4) * 100}%')
            log_print(f'SR = {total_success} / {count} = {round((total_success / count),4) * 100}%')
            log_print(f'SPL = {total_spl} / {count} = {round((total_spl / count),4) * 100}%')
            log_print('detail:')
            for k, v in reason_map.items():
                log_print(f'[{k}]:{v}')
            log_print('##########################')

        # write log content to file
        log_file_path = os.path.join(self.lmdb_path, 'eval.log')
        log_file_path1 = f'{PROJECT_ROOT_PATH}/logs/{self.name}/eval_result.log'
        with open(log_file_path, 'w') as f:
            f.write('\n'.join(log_content))
        with open(log_file_path1, 'w') as f:
            f.write('\n'.join(log_content))

        self.database_read.close()
