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
import numpy as np
import math
from math import pi
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from respawnGoal import Respawn

class Env():
    def __init__(self, action_size):
        self.goal_x = 0
        self.goal_y = 0
        self.heading = 0
        self.current_goal_distance = None
        self.action_size = action_size
        self.initGoal = True
        self.get_goalbox = False
        self.position = Pose()
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.respawn_goal = Respawn()

    def getGoal(self):
        return [self.goal_x, self.goal_y]

    def getPosition(self):
        return [self.position.x, self.position.y]

    def getGoalDistace(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)
        return goal_distance

    def getOdometry(self, odom):
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)

        heading = goal_angle - yaw
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        self.heading = round(heading, 2)
        self.previous_goal_distance = self.current_goal_distance
        self.current_goal_distance = self.getGoalDistace()

        if self.previous_goal_distance ==None: # TODO is there a better way for the first call?
            self.previous_goal_distance = self.current_goal_distance

    def getState(self, scan):
        done = False
        min_range = 0.13
        scan_normalize_const = 3.5
        goal_distance_normalize = 2
        heading_normalize = math.pi

        heading = self.heading/heading_normalize
        current_goal_distance = self.current_goal_distance / goal_distance_normalize


        scan_range = []
        n_scan_ranges = len(scan.ranges)
        for i in range(n_scan_ranges):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(1)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i]/scan_normalize_const)

        obstacle_min_range = round(min(scan_range)/scan_normalize_const, 2)
        obstacle_angle = np.argmin(scan_range)/n_scan_ranges*2-1

        if min_range > min(scan_range) > 0:
            done = True

        if current_goal_distance < 0.2:
            self.get_goalbox = True

        return scan_range + [heading, current_goal_distance, obstacle_min_range, obstacle_angle], done

    def setReward(self, state, done, action):

        reward = 0.1 if (self.previous_goal_distance-self.current_goal_distance) >0 else -0.1

        if done:
            rospy.loginfo("Collision!!")
            reward = -200
            self.pub_cmd_vel.publish(Twist())

        if self.get_goalbox:
            rospy.loginfo("Goal!!")
            reward = 200
            self.pub_cmd_vel.publish(Twist())
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
            self.start_goal_distance = self.getGoalDistace()
            self.get_goalbox = False

        return reward

    def step(self, action):
        max_angular_vel = 1.5
        ang_vel = ((self.action_size - 1)/2 - action) * max_angular_vel * 0.5

        vel_cmd = Twist()
        vel_cmd.linear.x = 0.15
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        state, done = self.getState(data)
        reward = self.setReward(state, done, action)

        return np.asarray(state), reward, done

    def reset(self):
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        if self.initGoal:
            self.goal_x, self.goal_y = self.respawn_goal.getPosition()
            self.initGoal = False

        self.start_goal_distance = self.getGoalDistace()
        state, done = self.getState(data)

        return np.asarray(state)