#!/usr/bin/env python
import datetime
import json
import numpy as np
from logging_script import logger
import os


def setup_logger(title, n_state_vals, action_dim, goal_dim, save_to_db=False,
                 cfg_file_path='/root/cfg/db_usr_cfg.json'):
    int_keys = ["run_id", "episode", "step"]

    float_keys = ["from_state_" + str(n) for n in range(n_state_vals)]
    float_keys += ["to_state_" + str(n) for n in range(n_state_vals)]
    float_keys += ["action"]  # ["action_" + str(n) for n in range(1)]
    float_keys += ["goal_" + str(n) for n in range(goal_dim)]
    float_keys += ["position_" + str(n) for n in range(goal_dim)]
    float_keys += ["q_vals_" + str(n) for n in range(action_dim)]
    float_keys += ["Reward"]

    string_keys = ["Title"]

    bool_keys = ["Terminal"]

    time_keys = ["Timestamp"]

    keys_ = int_keys + float_keys + string_keys + bool_keys + time_keys

    dtypes_ = ["bigint" for _ in range(len(int_keys))] + \
              ["real" for _ in range(len(float_keys))] + \
              ["varchar" for _ in range(len(string_keys))] + \
              ["boolean" for _ in range(len(bool_keys))] + \
              ["timestamp" for _ in range(len(time_keys))]

    if save_to_db and os.path.isfile(cfg_file_path):
        with open(cfg_file_path) as json_file:
            db_usr_cfg = json.load(json_file)

        db_config = {
            "database": {
                "host": db_usr_cfg["host"],
                "user": db_usr_cfg["user"],
                "passwd": db_usr_cfg["passwd"],
                "database": db_usr_cfg["database"],
                "port": db_usr_cfg["port"]
            },
            "schema_name": "lwidowski",
            "table_name": "tb_b_" + title,
            "key_list": keys_,
            "dtype_list": dtypes_,
            "primary_key": None,
            "auto_increment": None
        }

    else:
        db_config = None
        save_to_db = False

    log = logger(title=title + ".log", keys=keys_, dtypes=dtypes_, sep="\t", load_full=False, save_to_db=save_to_db,
                 db_config=db_config)
    return log, keys_


def make_log_entry(log, title, run_id, episode_number,
                   episode_step, from_state, to_state, goal, position,
                   action, q_vals,
                   reward, terminal):
    int_vals = [run_id, episode_number, episode_step]

    float_vals = np.asarray(from_state).flatten().tolist()
    float_vals += np.asarray(to_state).flatten().tolist()
    float_vals += [action]  # action.flatten().tolist()
    float_vals += np.asarray(goal).flatten().tolist()
    float_vals += np.asarray(position).flatten().tolist()
    float_vals += np.asarray(q_vals).flatten().tolist()
    float_vals += [reward]

    string_vals = [title]

    bool_vals = [terminal]

    time_vals = [str(datetime.datetime.now())]

    vals = int_vals + float_vals + string_vals + bool_vals + time_vals
    log.write_line(vals)
