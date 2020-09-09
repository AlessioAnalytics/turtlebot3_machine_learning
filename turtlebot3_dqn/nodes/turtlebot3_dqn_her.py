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

# Authors: Gilbert #

import rospy
import os
import json
import numpy as np
import time
import sys

from utils import log_utils
from dqn_her_agent import ReinforceAgent

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from std_msgs.msg import Float32MultiArray
from importlib import import_module


def save_model(agent, param_dictionary):
    agent.model.save(agent.dirPath + str(episode_number) + '.h5')
    with open(agent.dirPath + str(episode_number) + '.json', 'w') as outfile:
        json.dump(param_dictionary, outfile)
    agent.model.save(agent.dirPath + "latest" + '.h5')
    with open(agent.dirPath + "latest" + '.json', 'w') as outfile:
        json.dump(param_dictionary, outfile)
        print("MODEL SAVED", param_dictionary)


def get_time_since_start(start_time):
    minutes, seconds = divmod(int(time.time() - start_time), 60)
    hours, minutes = divmod(minutes, 60)
    return hours, minutes, seconds


def log_episode_info(episode_number, score, agent):
    hours, minutes, seconds = get_time_since_start(start_time)
    rospy.loginfo('Ep: %d score: %.2f memory: %d epsilon: %.2f time: %d:%02d:%02d',
                  episode_number, score, agent.her.n_entrys, agent.epsilon, hours, minutes, seconds)


def run_episode(env, global_step, param_dictionary):
    result = Float32MultiArray()
    get_action = Float32MultiArray()

    state = env.reset()
    goal = env.getGoal()
    score = 0

    for episode_step in range(agent.episode_max_steps):
        action = agent.get_action(state, goal)

        next_state, reward, done = env.step(action)
        her_goal = env.getPosition()
        agent.her.append_episode_replay(state, action, goal, her_goal, reward, next_state, done)
        log_utils.make_log_entry(log, log_title, run_id, episode_number,
                                 episode_step, state, next_state, goal, her_goal,
                                 action, agent.q_values,
                                 reward, done)

        if agent.her.n_entrys >= agent.train_start:
            if global_step <= agent.target_update:
                agent.train_model()
            else:
                agent.train_model(target=True)

        score += reward
        state = next_state
        get_action.data = [action, score, reward]
        pub_get_action.publish(get_action)

        if episode_number % 10 == 0 and episode_step == 0:
            save_model(agent, param_dictionary)

        if episode_step >= 500:
            rospy.loginfo("Time out!!")
            done = True

        if done:
            result.data = [score, np.max(agent.q_values)]
            pub_result.publish(result)
            agent.update_target_model()
            scores.append(score)
            episodes.append(episode_number)
            log_episode_info(episode_number, score, agent)

            param_keys = ['epsilon', 'episode']
            param_values = [agent.epsilon, episode_number]
            param_dictionary = dict(zip(param_keys, param_values))

            return global_step, param_dictionary

        global_step += 1
        if global_step % agent.target_update == 0:
            rospy.loginfo("UPDATE TARGET NETWORK")


if __name__ == '__main__':
    stage = rospy.get_param("/turtlebot3_dqn/stage")
    Env = import_module("src.turtlebot3_dqn.environment_stage_" + stage)
    rospy.init_node('turtlebot3_dqn_stage_' + stage)
    EPISODES = 3000

    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)

    state_size = 28
    action_size = 5
    goal_size = 2

    run_id = int(time.time())
    log_title = "turtlebot3_position"
    log, keys = log_utils.setup_logger(log_title, state_size, action_size, goal_dim=goal_size)
    env = Env.Env(action_size)
    agent = ReinforceAgent(state_size, action_size, goal_size, stage)

    scores, episodes = [], []
    global_step = 0
    start_time = time.time()
    param_keys = ['epsilon', 'episode']
    param_values = [agent.epsilon, agent.load_episode]
    param_dictionary = dict(zip(param_keys, param_values))

    for episode_number in range(agent.load_episode + 1, EPISODES):
        global_step, param_dictionary = run_episode(env, global_step, param_dictionary)

        agent.her.import_episode()
        log.save(save_to_db=True)
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
