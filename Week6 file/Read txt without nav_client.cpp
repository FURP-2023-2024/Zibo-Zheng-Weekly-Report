#include <ros/ros.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <actionlib/client/simple_action_client.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <thread>
#include <chrono>

typedef actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> MoveBaseClient;

std::string readLastLine(const std::string& filename) {
    std::ifstream file(filename.c_str());
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file");
    }

    std::string lastLine;
    std::string line;
    while (std::getline(file, line)) {
        lastLine = line;
    }
    file.close();
    return lastLine;
}

std::vector<int> parseLastLineToNumbers(const std::string& lastLine) {
    std::istringstream iss(lastLine);
    std::vector<int> numbers;
    int num;

    while (iss >> num) {
        numbers.push_back(num);
    }

    if (numbers.size() != 4) {
        throw std::runtime_error("The last line does not contain exactly four numbers");
    }

    return numbers;
}

void monitorFile(const std::string& filename, int intervalSeconds) {
    std::streampos lastSize = 0;

    while (true) {
        std::ifstream file(filename.c_str());
        if (!file.is_open()) {
            std::cerr << "Could not open file: " << filename << std::endl;
        } 
        else {
            file.seekg(0, std::ios::end);
            std::streampos newSize = file.tellg();

            if (newSize != lastSize) {
                lastSize = newSize;
                try {
                    std::string lastLine = readLastLine(filename);
                    // std::cout << "Last line: " << lastLine << std::endl; // 打印最后一行

                    std::vector<int> numbers = parseLastLineToNumbers(lastLine);
                    std::cout << "File size changed. Last line numbers: ";
                    for (int num : numbers) {
                        std::cout << num << ",";
                     }
                    std::cout << std::endl;
                } 
                catch (const std::exception& e) {
                    std::cerr << e.what() << std::endl;
                }
            }
        file.close();
    }

    std::this_thread::sleep_for(std::chrono::seconds(intervalSeconds));
    }
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "nav_client");

    std::string filename = "/home/admin123/catkin_ws/src/nav_pkg/src/your_file.txt"; // 替换为你的文件路径
    int intervalSeconds = 5; // Check interval in seconds

    std::cout << "Monitoring file: " << filename << " for changes..." << std::endl;
    monitorFile(filename, intervalSeconds);
   MoveBaseClient ac("move_base",true);//与move节点action通信

    while(!ac.waitForServer(ros::Duration(5.0))){
        ROS_INFO("Waiting for the move_base action server to come up");
    }

    move_base_msgs::MoveBaseGoal goal;
    goal.target_pose.header.frame_id="map";
    // goal.target_pose.header.stamp=ros::Time::now();

    goal.target_pose.pose.position.x=-1;
    goal.target_pose.pose.position.y=0;
    goal.target_pose.pose.position.z=0;
    goal.target_pose.pose.orientation.x=0;
    goal.target_pose.pose.orientation.w=1;
    goal.target_pose.pose.orientation.y=0;
    goal.target_pose.pose.orientation.z=0;

    // for(int i =0 ;i<=100;i++)
    // {
  ROS_INFO("Sending goal");
//  pub.publish(goal);
    ac.sendGoal(goal);
    ROS_INFO("Sending goal successfully");

    // }

    ac.waitForResult();
    if(ac.getState()==actionlib::SimpleClientGoalState::SUCCEEDED){
        ROS_INFO("Mission complete!");
    }
    else{
        ROS_INFO("Mission failed ...");
    }

    return 0;
}