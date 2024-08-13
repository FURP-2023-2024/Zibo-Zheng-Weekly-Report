# Zibo-Zheng-Weekly-Report
## Zibo Zheng Weekly Report

### Week 1     
#### 6.10-6.16
1. Find lidar models and sizes. Completed initial model of holder with 4d lidar and millimeter wave lidar.<br>
  ![](https://github.com/ZEbirds/Zibo-Zheng-Weekly-Report/blob/main/Holder1.png)

2. Install ubuntu 20.04 system (neotic) in a virtual machine and install ROS in it. <br>
Install vscode and configure the compilation environment. <br>
Install basic packages like rqt rviz gazebo etc.<br>

### Week 2     
#### 6.17-6.23
1. Update the version of Solidworks to 2024 so that we can receive the initial 4D lidar holder from project manager.<br>

2. Finished an advance model of holder with 4d lidar and millimeter wave lidar and connect it to the whole model of car and the aluminum profiles.<br>
   https://github.com/ZEbirds/Zibo-Zheng-Weekly-Report/tree/main/Holder%202<br>
   
3. Completing the initial study of ROS:<br>
      Learning the communication relationship between node and package and between packages and creating a node and package that can post topics to receive topics.<br>
      Utilizing c++ and python to implement functional nodes respectively.<br>
      Write some functional nodes such as implement robot motion control node, LIDAR data acquisition node (rviz observation radar data), LIDAR obstacle avoidance node, IMU data acquisition node, IMU heading lock node, etc.<br>
      These packages can be found in:<br>
https://github.com/ZEbirds/Zibo-Zheng-Weekly-Report/blob/main/Week2_pkgs.zip<br>
### Week 3     
#### 6.24-6.30
1. Finish the final model build of lidar holder.<br>
   https://github.com/ZEbirds/Zibo-Zheng-Weekly-Report/tree/main/Holder%20Final<br>
2. Advanced study of ROS:<br>
   <1>Understand and experience the Navigation system, move_base node, global planning, and AMCL positioning algorithm.<br>
     Understand the meaning of Costmap and parameterize the Costmap.<br>
     Understand the recovery behavior and parameterize the recovery behavior.<br>

   <2>Understand local planners including DWA planner and TEB planner and Action programming interface for navigation.<br>
     Write nav_client node using c++ and python and combine it with move_base for coordinate navigation.<br>
     Integrate and launch the waypoint navigation plugin waterplus_map_tools and write wp_node in c++ and python to implement the functionality to publish navigation points with wp_navi_server and receive navigation results.<br>

   <3>Learn about the camera topic in ROS and use c++ to acquire camera images.<br>
   These packages can be found in:<br>
   https://github.com/ZEbirds/Zibo-Zheng-Weekly-Report/blob/main/Week3_pkgs.zip<br>

### Week 4     
#### 7.1-7.7
Installed the unitree4d radar SDK and driver in my VM and configured the interface with the topic name so that the lidar point cloud of the actual environment could be successfully displayed in rviz.<br>
  ![](https://github.com/ZEbirds/Zibo-Zheng-Weekly-Report/blob/main/Pointcloud%20for%20unitree%20lidar.png)<br>

### Week 5     
#### 7.8-7.14
7.9<br>
Learn how to run functional packages such as cartographer and 3D point cloud maps to 2D maps in autolabor's mainframe (for testing offline map building and positioning)<br><br>
![](https://github.com/ZEbirds/Zibo-Zheng-Weekly-Report/blob/main/Week5%20file/cartographer%20in%20autolabor.jpg)<br><br>
![](https://github.com/ZEbirds/Zibo-Zheng-Weekly-Report/blob/main/Week5%20file/cartographer%20in%20autolabor1.jpg)<br><br>
![](https://github.com/ZEbirds/Zibo-Zheng-Weekly-Report/blob/main/Week5%20file/cartographer%20in%20autolabor2.jpg)<br><br>
![](https://github.com/ZEbirds/Zibo-Zheng-Weekly-Report/blob/main/Week5%20file/cartographer%20in%20autolabor3.jpg)<br><br>
![](https://github.com/ZEbirds/Zibo-Zheng-Weekly-Report/blob/main/Week5%20file/cartographer%20in%20autolabor4.jpg)<br><br>
7.10<br>
Manually calibrate the TF coordinates of the Lidar and the camera, and combine the depth camera information to overlay the two maps.<br><br>
![](https://github.com/ZEbirds/Zibo-Zheng-Weekly-Report/blob/main/Week5%20file/Manually%20calibrate.jpg)<br><br>
![](https://github.com/ZEbirds/Zibo-Zheng-Weekly-Report/blob/main/Week5%20file/Manually%20calibrate1.jpg)<br><br>
![](https://github.com/ZEbirds/Zibo-Zheng-Weekly-Report/blob/main/Week5%20file/Manually%20calibrate2.jpg)<br><br>
![](https://github.com/ZEbirds/Zibo-Zheng-Weekly-Report/blob/main/Week5%20file/Manually%20calibrate3.jpg)<br><br>
![](https://github.com/ZEbirds/Zibo-Zheng-Weekly-Report/blob/main/Week5%20file/Manually%20calibrate4.jpg)<br><br>
![](https://github.com/ZEbirds/Zibo-Zheng-Weekly-Report/blob/main/Week5%20file/Manually%20calibrate5.jpg)<br><br>
![](https://github.com/ZEbirds/Zibo-Zheng-Weekly-Report/blob/main/Week5%20file/Manually%20calibrate6.jpg)<br><br>
7.11<br> 
Install the lidar driver and camera driver on the host machine of autolabor.<br>
Convert the 4*4 rotation matrix to rpy Euler angles for outer parameter setup.<br><br>
7.12<br>
Reuse inverse matrix for rpy angles. Connect the lidar and depth camera online and run cartographer to locate and build the map.<br><br>
![](https://github.com/ZEbirds/Zibo-Zheng-Weekly-Report/blob/main/Week5%20file/rpy.png)<br><br>
7.13<br>
Install cartographer in my VM and run the official package for map building and localization testing, the results are as follows.<br><br>
![](https://github.com/ZEbirds/Zibo-Zheng-Weekly-Report/blob/main/Week5%20file/cartographer%20in%20my%20VM1.jpg)<br><br>
7.14<br>
Install autolabor control on the main machine of autolabor and download movebase to write the launch file and configure the global planner local planner and other parameters. Make autolabor can basically realize point-to-point navigation.<br><br>
![](https://github.com/ZEbirds/Zibo-Zheng-Weekly-Report/blob/main/Week5%20file/autolabor1.jpg)<br><br>
![](https://github.com/ZEbirds/Zibo-Zheng-Weekly-Report/blob/main/Week5%20file/SLAM%20in%20autolabor.jpg)<br><br>

### Week 6
#### 7.15-7.21
7.15<br>
Complete the static conversion(tf) from lidar(livox_frame) to base link.<br>
Configuration of vehicle shape parameters, target point toerance and expansion. radius<br>
Tuning point-to-point navigation.<br>
![](https://github.com/ZEbirds/Zibo-Zheng-Weekly-Report/blob/main/Week6%20file/move_base%20parameter1.jpg)<br><br>
![](https://github.com/ZEbirds/Zibo-Zheng-Weekly-Report/blob/main/Week6%20file/move_base%20parameter2.jpg)<br><br>
7.16<br>
Complete the static conversion of radar and camera tf, convert the camera point cloud to map tf for observation (for image recognition matching) and record the point cloud package.<br><br>
Write a navigation package that communicates with move_base to publish navigation points and receive navigation results.<br><br>
In the navigation package to realize the real-time reading and updating of navigation points in txt.<br><br>
Codes can be found in:<br>
https://github.com/FURP-2023-2024/Zibo-Zheng-Weekly-Report/blob/main/Week6%20file/nav_client.cpp<br>
https://github.com/FURP-2023-2024/Zibo-Zheng-Weekly-Report/blob/main/Week6%20file/Read%20txt%20with%20nav_client.cpp<br>
https://github.com/FURP-2023-2024/Zibo-Zheng-Weekly-Report/blob/main/Week6%20file/Read%20txt%20without%20nav_client.cpp<br><br>
7.17<br>
Navigation point sending and receiving between different robots over the same WLAN. <br>
Implementation of autolabor to send target navigation points to saite.<br><br>
The http send and receive functionality package can be found below<br>
https://github.com/FURP-2023-2024/Zibo-Zheng-Weekly-Report/blob/main/Week6%20file/httpROS.zip<br><br>
7.18<br>
Install conceptgraph and maskcluster and improve the algorithm (change the strategy to increase its robustness)<br>
The main code can be found in:<br><br>
https://github.com/FURP-2023-2024/Zibo-Zheng-Weekly-Report/blob/main/Week6%20file/process_pc_new_front-sim-real_first_floor.py<br><br>
Tuning the txt communication between maskcluster and nav_client and transferring navigation points.<br><br>
Write speed publish node to make the car rotate while travelling to the target point to store pictures of the surroundings. And publish topics to communicate with rosbagsaver to make it store photos. These photos will be compared or updated to the map when the target point is reached.<br><br>
![](https://github.com/FURP-2023-2024/Zibo-Zheng-Weekly-Report/blob/main/Week6%20file/READ%20ME.jpg)<br><br>
7.19<br>
Adjusting rotation speed to publish nodes and store photos<br><br>
Write navigation node and front end detection update object node using txt to communicate with robustnav and inform that both navigation and front end detection (using cv) are complete.The effect is as follows:<br>
![](https://github.com/FURP-2023-2024/Zibo-Zheng-Weekly-Report/blob/main/Week6%20file/maskcluster%20result%201.jpg)<br><br>
![](https://github.com/FURP-2023-2024/Zibo-Zheng-Weekly-Report/blob/main/Week6%20file/maskcluster%20result%202.jpg)<br><br>
![](https://github.com/FURP-2023-2024/Zibo-Zheng-Weekly-Report/blob/main/Week6%20file/maskcluster%20result%203.jpg)<br><br>
![](https://github.com/FURP-2023-2024/Zibo-Zheng-Weekly-Report/blob/main/Week6%20file/maskcluster%20result%204.jpg)<br><br>
Tuning through robustnav to detect plants in offline map and send navigation points to go to. It can be shown in the video below:<br>
https://github.com/FURP-2023-2024/Zibo-Zheng-Weekly-Report/blob/main/Week6%20file/VIDEO1.mp4<br><br>
7.21<br>
Replace the code for real-time detection with navigation to the target point completed and then front-end detect all the images together and change the import so that it communicates properly with maskcluster and nav_client<br><br>
The communication starts from when maskcluster releases the location of the target object searched from the offline map to navigating to the target location to cvmarking the pictures taken during the journey and updating the offline map, determining whether there is an object at the arrival location, and if there is not, then searching for the target object in the updated map and releasing the next target location. This process is shown in the figure below<br>
![](https://github.com/FURP-2023-2024/Zibo-Zheng-Weekly-Report/blob/main/Week6%20file/Process.jpg)<br><br>
The problem is not solved yet as shown in the figure below<br>
![](https://github.com/FURP-2023-2024/Zibo-Zheng-Weekly-Report/blob/main/Week6%20file/Challenges.jpg)<br><br>

### Week 7     
#### 7.22-7.28
7.22<br>
Rebuilt offline 2D map of new environment and 3D live map generated by rgb-d camera.<br>
![](https://github.com/FURP-2023-2024/Zibo-Zheng-Weekly-Report/blob/main/Week7%20file/rgb-d%20map1.jpg)<br><br>
![](https://github.com/FURP-2023-2024/Zibo-Zheng-Weekly-Report/blob/main/Week7%20file/rgb-d%20map2.jpg)<br><br>
![](https://github.com/FURP-2023-2024/Zibo-Zheng-Weekly-Report/blob/main/Week7%20file/rgb-d%20map3.jpg)<br><br>
Recorded the first demo to finish travelling to the target point to find the target (green plant) moved to a new location and travelling to the new target point<br><br>
7.23<br>
Discovered that the obstacle avoidance problem with the cart was due to the selection of the trajectoryplanner and modified the local path planner.<br><br>
Recorded further demos at different locations<br>
Video can be found in:<br>
https://github.com/FURP-2023-2024/Zibo-Zheng-Weekly-Report/blob/main/Week7%20file/demo%20video1.mp4<br><br>
Wrote a restart mechanism for maskcluster (front-end detection) so that it doesn't blow up memory the second time it runs.<br><br>
7.24<br>
Consider obstacle avoidance based on rotating nodes (rotating on the road and taking pictures with the camera to update the objects in the environment), by reading radar point clouds (pcl version is not compatible).<br><br>
Add the rotation node to record a demo.<br>
Try to record a demo of moving a target multiple times for target finding.<br><br>
Write a node that rotates multiple times (not yet in use)<br><br>
7.25<br>
Re-capture the camera map.<br>
![](https://github.com/FURP-2023-2024/Zibo-Zheng-Weekly-Report/blob/main/Week7%20file/camera%20map.jpg)<br><br>
Record two demos of books in different positions (with rotating observation update environment)<br><br>
Add rosbagsaver's self-restart mechanism so that each mission ends by clearing the photos and retaking them.<br>
Code can be found in https://github.com/FURP-2023-2024/Zibo-Zheng-Weekly-Report/blob/main/Week7%20file/imagetime_copy.cpp<br><br>
7.26<br>
Record the demo of the book for three tasks (multiple tasks) (with rotational observation to update the environment)<br>
The vedio can be found in https://github.com/FURP-2023-2024/Zibo-Zheng-Weekly-Report/blob/main/Week7%20file/demo%20video2.mp4<br><br>
Modify the semantic judgement of bearer.<br><br>
![](https://github.com/FURP-2023-2024/Zibo-Zheng-Weekly-Report/blob/main/Week7%20file/semantic%20judgement%20of%20bearer%201.jpg)<br><br>
![](https://github.com/FURP-2023-2024/Zibo-Zheng-Weekly-Report/blob/main/Week7%20file/semantic%20judgement%20of%20bearer%202.jpg)<br><br>
7.28<br>
Edit previously recorded demo video of first view third view fixed camera position<br><br>

### Week 8     
#### 7.29-8.4
7.29<br>
Cut demo video<br><br>
Begin to reproduce and validate loopback detection: there are three main steps, the first step is semantic segmentation of the laser point cloud, the second step is to add instance clustering, and the third step is to use the algorithm developed by the senior for loopback detection.<br><br>
Converting the actual point cloud bag from bit campus to bin file<br><br>

7.30-8.1<br>
Cylinder 3d model point cloud segmentation out of the label file and manipulate it with open3d visualisation (untrained model works poorly)<br><br>
Tried many point cloud segmentation networks like lidar-bonnetal, sphereformer, rangenet++, uniseg, cylinder3d, uniseg, cylinder3d, etc. (as below), none of them meet the demand.<br><br>
![](https://github.com/FURP-2023-2024/Zibo-Zheng-Weekly-Report/blob/main/Week8/segnet.jpg)<br><br>
8.2<br>
Convert the idea of trying to use the camera image segmentation and then mapped to the radar to achieve the radar point cloud segmentation (the camera view is limited and the point cloud is sparse).<br><br>
Eventually use the manual segmentation point cloud tool semantic-segmentation-editor to manually segment the point cloud<br>
![](https://github.com/FURP-2023-2024/Zibo-Zheng-Weekly-Report/blob/main/Week8/sse1.jpg)<br>
Write a python program to convert .bag to the desired .pcd format.<br>
Write python program to convert the processed pcd to .bin and .label files.<br>
Write scripts to visualise point clouds based on open3d.<br>
![](https://github.com/FURP-2023-2024/Zibo-Zheng-Weekly-Report/blob/main/Week8/sse-visualize.jpg)<br>
Write a python program to automatically align the timestamps of the bag and pose information, and filter out the frames needed for loopback detection.<br>
This py files can be found in https://github.com/FURP-2023-2024/Zibo-Zheng-Weekly-Report/blob/main/Week8/<br><br>
You can contact me for the specific packet<br><br>
8.3<br>
Manual segmentation of a 50-frame lidar point cloud<br><br>
8.4<br>
Manually split 58 frames of lidar point cloud<br><br>
### Week 9     
#### 8.5-8.11
8.5<br>
Manually split 59 frames of radar point cloud<br><br>
8.6<br>
Check 90 frames of point cloud labelled by other team members and modify them.<br><br>
Extract the coordinates and position of all point cloud frames. Code can be found below:<br>
https://github.com/FURP-2023-2024/Zibo-Zheng-Weekly-Report/blob/main/Week9%20file/readtime_final.py<br><br>
Extract every pair of distances less than 2 and more than 30 for subsequent detection of the loopback detection algorithm. Code can be found below:<br>
https://github.com/FURP-2023-2024/Zibo-Zheng-Weekly-Report/blob/main/Week9%20file/matchdata.py<br><br>
8.7<br>
End the BIT trip.<br><br>
Learn LOAM form blog:https://blog.csdn.net/gwplovekimi/article/details/119711762<br><br>
8.8<br>
Reusing the radar driver to acquire radar point cloud and imu data.<br><br>
8.9-8.10<br>
Use cartographer and a-loam to run online and offline slam builds respectively. The results are as follows:<br>
Online cartographer: ![](https://github.com/FURP-2023-2024/Zibo-Zheng-Weekly-Report/blob/main/Week9%20file/real-time%20cartographer.png)<br><br>
Offline a-loam: ![](https://github.com/FURP-2023-2024/Zibo-Zheng-Weekly-Report/blob/main/Week9%20file/offline%20aloam.png)<br><br>
Online a-loam: ![](https://github.com/FURP-2023-2024/Zibo-Zheng-Weekly-Report/blob/main/Week9%20file/online%20aloam.png)<br><br>
Online unilidarbag a-loam: ![](https://github.com/FURP-2023-2024/Zibo-Zheng-Weekly-Report/blob/main/Week9%20file/offline%20aloam(unilidar).png)<br><br>
The reason for the failure of aloam could be that it did not receive the imu data and used the icp to estimate the position directly, or that aloam is not good enough to deal with the single-line radar in the first place.<br><br>
8.11<br>
I have tried the algorithm lio-sam which combines the imu data in the loam series, but the offline construction has not reported any error and has not been run, while the online construction has reported "large velocity" because the imu internal parameter has not been set correctly. lio-sam algorithm needs to be further improved and tried.<br><br>
Read and understand the paper: SayNav: Grounding Large Language Models for Dynamic Planning to Navigation in New Environments and take notes.<br><br>
### Week 9     
#### 8.12-8.18
8.12<br>
Debug and retry lio-sam.<br>
![](https://github.com/FURP-2023-2024/Zibo-Zheng-Weekly-Report/blob/main/Week10%20file/imu%20error.png)<br><br>
![](https://github.com/FURP-2023-2024/Zibo-Zheng-Weekly-Report/blob/main/Week10%20file/imu%20error.png)<br><br>
Read and understand the paper: IVLMap: Instance-Aware Visual Language Grounding for Consumer Robot Navigation <br><br>
8.13<br>
Get the battery and start collecting lidar and imu packets to keep for debugging algorithms online.<br>
![](https://github.com/FURP-2023-2024/Zibo-Zheng-Weekly-Report/blob/main/Week10%20file/理工楼.jpeg)<br><br>
Debug the parameters of the loopback detection code and run a test of the previously obtained data pairs.<br>
![](https://github.com/FURP-2023-2024/Zibo-Zheng-Weekly-Report/blob/main/Week10%20file/loopback%20detection.jpeg)<br><br>
