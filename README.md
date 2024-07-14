# Zibo-Zheng-Weekly-Report
## Zibo Zheng Weekly Report

### Week 1     
#### 6.10-6.16
1. Find lidar models and sizes. Completed initial model of holder with 4d lidar and millimeter wave lidar.<br>
  ![](https://github.com/ZEbirds/Zibo-Zheng-Weekly-Report/blob/main/Holder1.png)

2. Install ubuntu 20.04 system (neotic) in a virtual machine and install ROS in it. <br>Install vscode and configure the compilation environment. <br>Install basic packages like rqt rviz gazebo etc.<br>

### Week 2     
#### 6.17-6.23
1. Update the version of Solidworks to 2024 so that we can receive the initial 4D lidar holder from project manager.<br>

2. Finished an advance model of holder with 4d lidar and millimeter wave lidar and connect it to the whole model of car and the aluminum profiles.<br>

3. Completing the initial study of ROS:<br>
      Learning the communication relationship between node and package and between packages and creating a node and package that can post topics to receive topics.<br>
      Utilizing c++ and python to implement functional nodes respectively.<br>
      Write some functional nodes such as implement robot motion control node, LIDAR data acquisition node (rviz observation radar data), LIDAR obstacle avoidance node, IMU data acquisition node, IMU heading lock node, etc.<br>
      These packages can be found in:<br>

### Week 3     
#### 6.24-6.30
1. Finish the final model build of lidar holder.<br>
2. Advanced study of ROS:<br>
   <1>Understand and experience the Navigation system, move_base node, global planning, and AMCL positioning algorithm.<br>
     Understand the meaning of Costmap and parameterize the Costmap.<br>
     Understand the recovery behavior and parameterize the recovery behavior.<br>

   <2>Understand local planners including DWA planner and TEB planner and Action programming interface for navigation.<br>
     Write nav_client node using c++ and python and combine it with move_base for coordinate navigation.<br>
     Integrate and launch the waypoint navigation plugin waterplus_map_tools and write wp_node in c++ and python to implement the functionality to publish navigation points with wp_navi_server and receive navigation results.<br>

   <3>Learn about the camera topic in ROS and use c++ to acquire camera images.<br>
   These packages can be found in:<br>

### Week 4     
#### 7.1-7.7
Installed the unitree4d radar SDK and driver in my VM and configured the interface with the topic name so that the lidar point cloud of the actual environment could be successfully displayed in rviz.<br>
  ![](https://github.com/ZEbirds/Zibo-Zheng-Weekly-Report/blob/main/Pointcloud%20for%20unitree%20lidar.png)

### Week 5     
#### 7.8-7.14
7.9 Learn how to run functional packages such as cartographer and 3D point cloud maps to 2D maps in autolabor's mainframe (for testing offline map building and positioning)<br><br>
![](https://github.com/ZEbirds/Zibo-Zheng-Weekly-Report/blob/main/Week5%20file/cartographer%20in%20autolabor.jpg)<br><br>
![](https://github.com/ZEbirds/Zibo-Zheng-Weekly-Report/blob/main/Week5%20file/cartographer%20in%20autolabor1.jpg)<br><br>
![](https://github.com/ZEbirds/Zibo-Zheng-Weekly-Report/blob/main/Week5%20file/cartographer%20in%20autolabor2.jpg)<br><br>
![](https://github.com/ZEbirds/Zibo-Zheng-Weekly-Report/blob/main/Week5%20file/cartographer%20in%20autolabor3.jpg)<br><br>
![](https://github.com/ZEbirds/Zibo-Zheng-Weekly-Report/blob/main/Week5%20file/cartographer%20in%20autolabor4.jpg)<br><br>
7.10 Manually calibrate the TF coordinates of the Lidar and the camera, and combine the depth camera information to overlay the two maps.<br><br>
![](https://github.com/ZEbirds/Zibo-Zheng-Weekly-Report/blob/main/Week5%20file/Manually%20calibrate.jpg)<br><br>
![](https://github.com/ZEbirds/Zibo-Zheng-Weekly-Report/blob/main/Week5%20file/Manually%20calibrate1.jpg)<br><br>
![](https://github.com/ZEbirds/Zibo-Zheng-Weekly-Report/blob/main/Week5%20file/Manually%20calibrate2.jpg)<br><br>
![](https://github.com/ZEbirds/Zibo-Zheng-Weekly-Report/blob/main/Week5%20file/Manually%20calibrate3.jpg)<br><br>
![](https://github.com/ZEbirds/Zibo-Zheng-Weekly-Report/blob/main/Week5%20file/Manually%20calibrate4.jpg)<br><br>
![](https://github.com/ZEbirds/Zibo-Zheng-Weekly-Report/blob/main/Week5%20file/Manually%20calibrate5.jpg)<br><br>
![](https://github.com/ZEbirds/Zibo-Zheng-Weekly-Report/blob/main/Week5%20file/Manually%20calibrate6.jpg)<br><br>
7.11 Install the lidar driver and camera driver on the host machine of autolabor.<br>
7.11 Convert the 4*4 rotation matrix to rpy Euler angles for outer parameter setup.<br><br>
7.12 Reuse inverse matrix for rpy angles. Connect the lidar and depth camera online and run cartographer to locate and build the map.<br><br>
7.13 Install cartographer in my VM and run the official package for map building and localization testing, the results are as follows.<br><br>
7.14 Install autolabor control on the main machine of autolabor and download movebase to write the launch file and configure the global planner local planner and other parameters. Make autolabor can basically realize point-to-point navigation.<br><br>
