sudo docker run -it --env="DISPLAY" --env="QT_X11_NO_MITSHM=1" -v "/home/lmueller/saved_models_own_architecture:/root/catkin_ws/src/turtlebot3_machine_learning/turtlebot3_dqn/save_model:rw" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" -v "/home/lmueller/turtlebot3_machine_learning/docker_setup/cfg:/root/cfg:rw" ros_ml