# TurtleBot3

Repo to run a Turtlebot3 Reinforcement Learning Office Environment.

This is built on the amazing work of the ROBOTIS Open Source Team. It is easier 
to run since it doesn't require a local ROS installation as we use Docker.

## How to use
Enable ports for GUI to run properly:

`xhost +local:root`


Start docker container:

`. docker_setup/my_start_container.sh`

In the container start simulation and agent:

`. /my_scripts/start_ros.sh $STAGE_PARAM`

where $STAGE_PARAM has to be replaced with a number from 1 to 4 depending on
the stage you want to start.

If no simulation is starting on your screen make sure `$DISPLAY` is set correctly.

If you run into problems with logging or database cfg files disable logging in
turtlebot3_dqn.py

## Tips 
### Speedup Simulation

In the Gazebo Gui on the left side on the world tab click Physics and set real time update to 0.
This will make Gazebo run as fast as possible.

## ROBOTIS Content for TurtleBot3
- [ROBOTIS e-Manual for TurtleBot3](http://turtlebot3.robotis.com/)
- [META-Package Documentation](http://wiki.ros.org/turtlebot3_machine_learning)
- [DQN-Package Documentation](http://wiki.ros.org/turtlebot3_dqn)
- [ROBOTIS e-Manual for TurtleBot3](http://turtlebot3.robotis.com/)
- [Website for TurtleBot Series](http://www.turtlebot.com/)
- [e-Book for TurtleBot3](https://community.robotsource.org/t/download-the-ros-robot-programming-book-for-free/51/)
- [Videos for TurtleBot3](https://www.youtube.com/playlist?list=PLRG6WP3c31_XI3wlvHlx2Mp8BYqgqDURU)
