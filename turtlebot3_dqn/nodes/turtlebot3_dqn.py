#!/usr/bin/env python
#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

# Authors: Gilbert, Widowski, Mueller #

import rospy
import os
import numpy as np
import time
import sys

from dqn_agent import ReinforceAgent
from utils import log_utils
from utils.model_utils import save_model, log_episode_info

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from std_msgs.msg import Float32MultiArray
from importlib import import_module


def run_episode(agent, env, pub_result, pub_get_action, run_id, episode_number,
                global_step, param_dictionary, start_time, scores, episodes,
                log, log_title, save_model_to_disk):
    result = Float32MultiArray()
    get_action = Float32MultiArray()

    state = env.reset()
    score = 0
    for episode_step in range(agent.episode_max_step):
        goal = env.get_goal()
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)

        if episode_step >= agent.episode_max_step - 1:
            rospy.loginfo("Time out!!")
            done = True

        position = env.get_position()
        agent.append_memory(state, action, reward, next_state, done)
        log_utils.make_log_entry(log, log_title, run_id, episode_number,
                                 episode_step, state, next_state, goal, position,
                                 action, agent.q_value, reward, done)

        if len(agent.memory) >= agent.train_start:
            if global_step <= agent.target_update:
                agent.train_model()
            else:
                agent.train_model(True)

        score += reward
        state = next_state
        get_action.data = [action, score, reward]
        pub_get_action.publish(get_action)

        if save_model_to_disk and episode_number % 10 == 0 and episode_step == 0:
            save_model(agent, param_dictionary, episode_number)

        if done:
            result.data = [score, np.max(agent.q_value)]
            pub_result.publish(result)
            agent.update_target_model()
            scores.append(score)
            episodes.append(episode_number)
            log_episode_info(episode_number, score, agent, start_time)

            param_keys = ['epsilon', 'episode']
            param_values = [agent.epsilon, episode_number]
            param_dictionary = dict(zip(param_keys, param_values))

            return run_id, global_step, param_dictionary

        global_step += 1
        if global_step % agent.target_update == 0:
            rospy.loginfo("UPDATE TARGET NETWORK")


if __name__ == '__main__':
    EPISODES = 1000

    stage = rospy.get_param("/turtlebot3_dqn/stage")
    Env = import_module("src.turtlebot3_dqn.environment")  # TODO change import to normal import
    rospy.init_node('turtlebot3_dqn_stage_' + stage)  # TODO is this necessary

    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)

    state_size = 28
    action_size = 5

    run_id = int(time.time())
    log_title = "turtlebot3_dqn"
    log, keys = log_utils.setup_logger(log_title, state_size, action_size, goal_dim=2, save_to_db=True)
    env = Env.Env(action_size)
    agent = ReinforceAgent(state_size, action_size, stage, episode_max_step=1000)

    scores, episodes = [], []
    global_step = 0
    start_time = time.time()
    param_keys = ['epsilon', 'episode']
    param_values = [agent.epsilon, agent.load_episode]
    param_dictionary = dict(zip(param_keys, param_values))

    for episode_number in range(agent.load_episode + 1, EPISODES):
        run_id, global_step, param_dictionary = run_episode(agent, env, pub_result=pub_result,
                                                            pub_get_action=pub_get_action, run_id=run_id,
                                                            global_step=global_step, param_dictionary=param_dictionary,
                                                            start_time=start_time, scores=scores, episodes=episodes,
                                                            log=log, log_title=log_title, episode_number=episode_number,
                                                            save_model_to_disk=True)

        log.save()

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
