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

# Authors: Mueller #

import numpy as np
import math
from math import pi


def punish_sparse(goal_reached):
    """
    from: Deep Reinforcement Learning with Successor Features
          for Navigation across Similar Environments
    """
    if goal_reached:
        return 1

    else:
        return -0.04


def punish_no_sparse(goal_reached, goal_distance):
    if goal_reached:
        return 100

    else:
        return -np.exp(goal_distance) / 100


def legacy_reward(state, done, action, start_goal_distance, goal_reached):
    """
    Legacy reward function in original Repo from ROBOTIS
    """
    yaw_reward = []
    current_goal_distance = state[-3]
    heading = state[-4]

    for i in range(5):
        angle = -pi / 4 + heading + (pi / 8 * i) + pi / 2
        tr = 1 - 4 * math.fabs(0.5 - math.modf(0.25 + 0.5 * angle % (2 * math.pi) / math.pi)[0])
        yaw_reward.append(tr)

    distance_rate = 2 ** (current_goal_distance / start_goal_distance)
    reward = ((round(yaw_reward[action] * 5, 2)) * distance_rate)

    if done:
        reward = -200
    if goal_reached:
        reward = 200

    return reward
