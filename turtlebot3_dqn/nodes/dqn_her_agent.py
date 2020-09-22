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
import json
import numpy as np
import random
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from hindsight_experience_replay import HindsightExperienceReplay
from std_msgs.msg import Float32MultiArray
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop
from keras.layers import Dense, Dropout, Activation


class ReinforceAgent:
    def __init__(self, state_size, action_size, goal_size, stage="1",
                 episode_max_steps=6000, target_update=2000, discount_factor=0.99,
                 learning_rate=0.00025, epsilon=1.0, epsilon_decay=0.99,
                 epsilon_min=0.05, batch_size=64, train_start=64, load_model_bool=True):
        self.pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
        self.dirPath = os.path.dirname(os.path.realpath(__file__))
        self.dirPath = self.dirPath.replace('turtlebot3_dqn/nodes', 'turtlebot3_dqn/save_model/stage_' + stage + '_')
        self.result = Float32MultiArray()

        self.load_episode = "latest"
        self.state_size = state_size
        self.action_size = action_size
        self.goal_size = goal_size
        self.episode_max_steps = episode_max_steps
        self.target_update = target_update
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.train_start = train_start

        self.her = HindsightExperienceReplay(k=1, strategy="future", maxlen=1000000, batch_size=self.batch_size)
        self.q_values = np.zeros(self.action_size)

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

        if load_model_bool:
            self.load_saved_model()

    def load_saved_model(self):
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

    def build_model(self):
        model = Sequential([
            Dense(64, input_shape=(self.state_size + self.goal_size), kernel_initializer='lecun_uniform'),
            Activation('relu'),
            Dense(64, kernel_initializer='lecun_uniform'),
            Activation('relu'),
            Dropout(0.2),
            Dense(self.action_size, kernel_initializer='lecun_uniform')
        ])

        model.compile(loss='mse', optimizer=RMSprop(lr=self.learning_rate, rho=0.9, epsilon=1e-06))
        model.summary()

        return model

    def get_q_value(self, reward, next_target, done):
        if done:
            return reward
        else:
            return reward + self.discount_factor * np.amax(next_target)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def predict(self, state, goal):
        features = np.hstack((np.asarray(state).flatten(), np.asarray(goal).flatten()))
        return self.model.predict(features.reshape(1, len(state) + len(goal)))

    def predict_target(self, state, goal):
        features = np.hstack((np.asarray(state).flatten(), np.asarray(goal).flatten()))
        return self.target_model.predict(features.reshape(1, len(state) + len(goal)))

    def get_action(self, state, goal):
        if np.random.rand() <= self.epsilon:
            self.q_values = np.zeros(self.action_size)
            return random.randrange(self.action_size)
        else:
            q_value = self.predict(state, goal)
            self.q_values = q_value
            return np.argmax(q_value[0])

    def train_model(self, target=False):
        mini_batch = self.her.sample_memory()
        X_batch = np.empty((0, self.state_size + self.goal_size), dtype=np.float64)
        Y_batch = np.empty((0, self.action_size), dtype=np.float64)

        for i in range(self.batch_size):
            states, actions, goals, rewards, next_states, dones = mini_batch[i]
            self.q_values = self.predict(states, goals)

            if target:
                next_target = self.predict_target(next_states, goals)

            else:
                next_target = self.predict(next_states, goals)

            next_q_values = self.get_q_value(rewards, next_target, dones)

            X_batch = np.append(X_batch, np.asarray([np.hstack((states, goals)).copy()]), axis=0)
            Y_sample = self.q_values.copy()

            Y_sample[0][actions] = next_q_values
            Y_batch = np.append(Y_batch, np.array([Y_sample[0]]), axis=0)

            if dones:
                X_batch = np.append(X_batch, np.asarray([np.hstack((next_states, goals)).copy()]), axis=0)
                Y_batch = np.append(Y_batch, np.array([[rewards] * self.action_size]), axis=0)

        self.model.fit(X_batch, Y_batch, batch_size=self.batch_size, epochs=1, verbose=0)
