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
import os
import sys
import json
import random
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from collections import deque
from std_msgs.msg import Float32MultiArray
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop
from keras.layers import Dense, Dropout, Activation, BatchNormalization


class ReinforceAgent:
    def __init__(self, state_size, action_size, stage="1",
                 episode_max_step=6000, target_update=2000, discount_factor=0.99,
                 learning_rate=0.00025, epsilon=1.0, epsilon_decay=0.99,
                 epsilon_min=0.05, batch_size=64, train_start=64, load_model_bool=True):
        self.pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
        self.dirPath = os.path.dirname(os.path.realpath(__file__))
        self.dirPath = self.dirPath.replace('turtlebot3_dqn/nodes', 'turtlebot3_dqn/save_model/stage_' + stage + '_')
        self.result = Float32MultiArray()

        self.load_model = load_model_bool
        self.load_episode = "latest"
        self.state_size = state_size
        self.action_size = action_size
        self.episode_max_step = episode_max_step
        self.target_update = target_update
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.train_start = train_start

        self.memory = deque(maxlen=1000000)
        self.q_value = np.zeros(self.action_size)
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

    def get_q_value(self, reward, next_target, done):
        if done:
            return reward
        else:
            return reward + self.discount_factor * np.amax(next_target)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            self.q_value = np.zeros(self.action_size)
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state.reshape(1, len(state)))
            self.q_value = q_value
            return np.argmax(q_value[0])

    def append_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_model(self, target=False):
        mini_batch = random.sample(self.memory, self.batch_size)
        X_batch = np.empty((0, self.state_size), dtype=np.float64)
        Y_batch = np.empty((0, self.action_size), dtype=np.float64)

        for i in range(self.batch_size):
            states, actions, rewards, next_states, dones = mini_batch[i]
            self.q_value = self.model.predict(states.reshape(1, len(states)))

            if target:
                next_target = self.target_model.predict(next_states.reshape(1, len(next_states)))

            else:
                next_target = self.model.predict(next_states.reshape(1, len(next_states)))

            next_q_value = self.get_q_value(rewards, next_target, dones)

            X_batch = np.append(X_batch, np.array([states.copy()]), axis=0)
            Y_sample = self.q_value.copy()

            Y_sample[0][actions] = next_q_value
            Y_batch = np.append(Y_batch, np.array([Y_sample[0]]), axis=0)

            if dones:
                X_batch = np.append(X_batch, np.array([next_states.copy()]), axis=0)
                Y_batch = np.append(Y_batch, np.array([[rewards] * self.action_size]), axis=0)

        self.model.fit(X_batch, Y_batch, batch_size=self.batch_size, epochs=1, verbose=0)
