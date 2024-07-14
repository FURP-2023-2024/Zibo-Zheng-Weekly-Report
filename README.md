# Zibo-Zheng-Weekly-Report
## Zibo Zheng Weekly Report

### Week 1 6.10-6.16
1. Find lidar models and sizes. Completed initial model of holder with 4d lidar and millimeter wave lidar.
  ![](https://github.com/ZEbirds/Zibo-Zheng-Weekly-Report/blob/main/Holder1.png)

2. Install ubuntu 20.04 system (neotic) in a virtual machine and install ROS in it. Install vscode and configure the compilation environment. Install basic packages like rqt rviz gazebo etc.

### Week 2 6.17-6.23
1. Update the version of Solidworks to 2024 so that we can receive the initial 4D lidar holder from project manager.
2. Finished an advance model of holder with 4d lidar and millimeter wave lidar and connect it to the whole model of car and the aluminum profiles.
3. Completing the initial study of ROS:
      Learning the communication relationship between node and package and between packages and creating a node and package that can post topics to receive topics.
      Utilizing c++ and python to implement functional nodes respectively.
      Write some functional nodes such as implement robot motion control node, LIDAR data acquisition node (rviz observation radar data), LIDAR obstacle avoidance node, IMU data acquisition node, IMU heading lock node, etc. These packages can be found in:

### Week 3 6.24-6.30
1. Finish the final model build of lidar holder.
2. Advanced study of ROS:
   <1>Understand and experience the Navigation system, move_base node, global planning, and AMCL positioning algorithm.
      Understand the meaning of Costmap and parameterize the Costmap.
      Understand the recovery behavior and parameterize the recovery behavior.

   <2>Understand local planners including DWA planner and TEB planner and Action programming interface for navigation.
      Write nav_client node using c++ and python and combine it with move_base for coordinate navigation.
      Integrate and launch the waypoint navigation plugin waterplus_map_tools and write wp_node in c++ and python to implement the functionality to publish navigation points with wp_navi_server and receive navigation results.

   <3>Learn about the camera topic in ROS and use c++ to acquire camera images.
   These packages can be found in:
