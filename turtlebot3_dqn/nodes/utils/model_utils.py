#!/usr/bin/env python
#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
# Copyright 2020 Alessio Analytics GmbH
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
import json
import time


def log_episode_info(episode_number, score, agent, start_time):
    hours, minutes, seconds = get_time_since_start(start_time)
    rospy.loginfo('Ep: %d score: %.2f memory: %d epsilon: %.2f time: %d:%02d:%02d',
                  episode_number, score, len(agent.memory), agent.epsilon, hours, minutes, seconds)


def save_model(agent, param_dictionary, episode_number):
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
