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
import random
import time
import sys

from utils import log_utils

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from collections import deque
from std_msgs.msg import Float32MultiArray
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from importlib import import_module

EPISODES = 3000


class ReinforceAgent():
    def __init__(self, state_size, action_size, stage="1"):
        self.pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
        self.dirPath = os.path.dirname(os.path.realpath(__file__))
        self.dirPath = self.dirPath.replace('turtlebot3_dqn/nodes', 'turtlebot3_dqn/save_model/stage_' + stage + '_')
        self.result = Float32MultiArray()

        self.load_model = True
        self.load_episode = "latest"
        self.state_size = state_size
        self.action_size = action_size
        self.episode_step = 6000
        self.target_update = 2000
        self.discount_factor = 0.99
        self.learning_rate = 0.00025
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.05
        self.batch_size = 64
        self.train_start = 64
        self.memory = deque(maxlen=1000000)

        self.model = self.buildModel()
        self.target_model = self.buildModel()

        self.updateTargetModel()
        if self.load_model:
            if not os.path.isfile(self.dirPath + str(self.load_episode) + ".h5"):
                print("file: ", str(self.dirPath + str(self.load_episode) + ".h5"), "is not present!")
                print("continue with randomly initialized model")
                self.load_episode = 0
            else:
                self.model.set_weights(load_model(self.dirPath + str(self.load_episode) + ".h5").get_weights())

                with open(self.dirPath + str(self.load_episode) + '.json') as outfile:
                    param = json.load(outfile)

                self.epsilon = param.get('epsilon')
                self.load_episode = param.get('episode')

                print("latest model restored")
                print("previously trained for", str(self.load_episode), "episodes")

    def buildModel(self):
        model = Sequential([
                Dense(64, input_shape=(self.state_size,), kernel_initializer='lecun_uniform'),
                Activation('relu'),
                Dense(64, kernel_initializer='lecun_uniform'),
                Activation('relu'),
                Dropout(0.2),
                Dense(self.action_size, kernel_initializer='lecun_uniform'),
        ])
        model.compile(loss='mse', optimizer=RMSprop(lr=self.learning_rate, rho=0.9, epsilon=1e-06))
        model.summary()

        return model

    def getQvalue(self, reward, next_target, done):
        if done:
            return reward
        else:
            return reward + self.discount_factor * np.amax(next_target)

    def updateTargetModel(self):
        self.target_model.set_weights(self.model.get_weights())

    def getAction(self, state):
        if np.random.rand() <= self.epsilon:
            self.q_value = np.zeros(self.action_size)
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state.reshape(1, len(state)))
            self.q_value = q_value
            return np.argmax(q_value[0])

    def appendMemory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def trainModel(self, target=False):
        mini_batch = random.sample(self.memory, self.batch_size)
        X_batch = np.empty((0, self.state_size), dtype=np.float64)
        Y_batch = np.empty((0, self.action_size), dtype=np.float64)

        for i in range(self.batch_size):
            states = mini_batch[i][0]
            actions = mini_batch[i][1]
            rewards = mini_batch[i][2]
            next_states = mini_batch[i][3]
            dones = mini_batch[i][4]

            q_value = self.model.predict(states.reshape(1, len(states)))
            self.q_value = q_value

            if target:
                next_target = self.target_model.predict(next_states.reshape(1, len(next_states)))

            else:
                next_target = self.model.predict(next_states.reshape(1, len(next_states)))

            next_q_value = self.getQvalue(rewards, next_target, dones)

            X_batch = np.append(X_batch, np.array([states.copy()]), axis=0)
            Y_sample = q_value.copy()

            Y_sample[0][actions] = next_q_value
            Y_batch = np.append(Y_batch, np.array([Y_sample[0]]), axis=0)

            if dones:
                X_batch = np.append(X_batch, np.array([next_states.copy()]), axis=0)
                Y_batch = np.append(Y_batch, np.array([[rewards] * self.action_size]), axis=0)

        self.model.fit(X_batch, Y_batch, batch_size=self.batch_size, epochs=1, verbose=0)


if __name__ == '__main__':
    stage = rospy.get_param("/turtlebot3_dqn/stage")

    Env = import_module("src.turtlebot3_dqn.environment_stage_" + stage)

    rospy.init_node('turtlebot3_dqn_stage_'+stage)

    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)

    result = Float32MultiArray()
    get_action = Float32MultiArray()

    state_size = 28
    action_size = 5

    run_id = int(time.time())
    log_title = "turtlebot3_dqn"
    log, keys = log_utils.setup_logger(log_title, state_size, action_size, goal_dim=2)
    env = Env.Env(action_size)
    agent = ReinforceAgent(state_size, action_size, stage)

    scores, episodes = [], []
    global_step = 0
    start_time = time.time()
    param_keys = ['epsilon', 'episode']
    param_values = [agent.epsilon, agent.load_episode]
    param_dictionary = dict(zip(param_keys, param_values))

    for episode_number in range(agent.load_episode + 1, EPISODES):
        done = False
        state = env.reset()
        goal = env.getGoal()
        score = 0
        start_goal = env.getGoal()
        for episode_step in range(agent.episode_step):
            goal = env.getGoal()
            action = agent.getAction(state)

            next_state, reward, done = env.step(action)

            if episode_step >= 500:
                rospy.loginfo("Time out!!")
                if goal == start_goal:
                    reward = -200
                done = True

            position = env.getPosition()
            agent.appendMemory(state, action, reward, next_state, done)
            log_utils.make_log_entry(log, log_title, run_id, episode_number,
                                     episode_step, state, next_state, goal, position,
                                     action, agent.q_value,
                                     reward, done)

            if len(agent.memory) >= agent.train_start:
                if global_step <= agent.target_update:
                    agent.trainModel()
                else:
                    agent.trainModel(True)

            score += reward
            state = next_state
            get_action.data = [action, score, reward]
            pub_get_action.publish(get_action)

            if episode_number % 10 == 0 and episode_step == 0:
                agent.model.save(agent.dirPath + str(episode_number) + '.h5')
                with open(agent.dirPath + str(episode_number) + '.json', 'w') as outfile:
                    json.dump(param_dictionary, outfile)

                agent.model.save(agent.dirPath + "latest" + '.h5')
                with open(agent.dirPath + "latest" + '.json', 'w') as outfile:
                    json.dump(param_dictionary, outfile)
                    print("MODEL SAVED", param_dictionary)

            if done:
                result.data = [score, np.max(agent.q_value)]
                pub_result.publish(result)
                agent.updateTargetModel()
                scores.append(score)
                episodes.append(episode_number)
                m, s = divmod(int(time.time() - start_time), 60)
                h, m = divmod(m, 60)

                rospy.loginfo('Ep: %d score: %.2f memory: %d epsilon: %.2f time: %d:%02d:%02d',
                              episode_number, score, len(agent.memory), agent.epsilon, h, m, s)
                param_keys = ['epsilon', 'episode']
                param_values = [agent.epsilon, episode_number]
                param_dictionary = dict(zip(param_keys, param_values))
                break

            global_step += 1
            if global_step % agent.target_update == 0:
                rospy.loginfo("UPDATE TARGET NETWORK")

        log.save(save_to_db=True)
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

