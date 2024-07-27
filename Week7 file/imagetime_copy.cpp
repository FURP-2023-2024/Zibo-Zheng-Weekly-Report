#include <ros/ros.h>
#include <std_msgs/Int32.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2_msgs/TFMessage.h>
#include <nav_msgs/Odometry.h>
#include <tf2_ros/transform_listener.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <boost/filesystem.hpp>
#include <Eigen/Geometry>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <thread>
#include <chrono>

// namespace fs = std::filesystem;
namespace bf = boost::filesystem;

int i = 0;
int rotate_flag =1 ;

ros::Time timestampold = ros::Time(0);
ros::Time timestampnow;

std::string newobv_storage_path = "/home/admin123/robustnav/data/new_obv/";
std::string rgb_storage_path = "/home/admin123/robustnav/data/new_obv/color/";
std::string depth_storage_path = "/home/admin123/robustnav/data/new_obv/depth/";
std::string transform_storage_path = "/home/admin123/robustnav/data/new_obv/pose/";
std::string filename = "/home/admin123/catkin_ws/src/nav_pkg/src/your_file.txt"; 
// file.seekg(0, std::ios::end);
std::streampos lastSize =0 ;
std::streampos newSize =0;
// 回调函数，当接收到消息时调用
void rotate_flag_Callback(const std_msgs::Int32::ConstPtr& msg)
{
    ROS_INFO("Received: %d", msg->data);
    rotate_flag = msg->data;
}

void callback(const sensor_msgs::ImageConstPtr& depth_msg,
              const sensor_msgs::ImageConstPtr& rgb_msg,
              const nav_msgs::Odometry::ConstPtr& tf_msg1)
{
         std::cout<<"11111111111111"<<std::endl;
        //  printf("lastSize: %ld .....\n", static_cast<long>(lastSize));
     std::ifstream file(filename.c_str());
if (!file.is_open()) {
    std::cerr << "Could not open file: " << filename << std::endl;
} else {
    file.seekg(0, std::ios::end);
    newSize = file.tellg();
    printf("lastSize: %ld, newSize: %ld\n", static_cast<long>(lastSize), static_cast<long>(newSize));
    if (newSize != lastSize) {
        lastSize = newSize;
        printf("restart!!!\n");
        fflush(stdout);
        char *args[] = {"/home/admin123/rosbagsaver/build/devel/lib/rosbagsaver/rosbagsaver", NULL};
        execvp(args[0], args);
        perror("execvp failed");
        // system("/home/admin123/rosbagsaver/build/devel/lib/rosbagsaver/rosbagsaver");
    }
            else{
   
    // 将 ROS 图像消息转换为 OpenCV 格式
    cv_bridge::CvImagePtr rgb_cv_ptr = cv_bridge::toCvCopy(rgb_msg, sensor_msgs::image_encodings::BGR8);
    cv::Mat rgb_resized;
    cv::resize(rgb_cv_ptr->image, rgb_resized, cv::Size(768, 576), cv::INTER_CUBIC);

    cv_bridge::CvImagePtr depth_cv_ptr = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
    cv::Mat depth_resized;
    cv::resize(depth_cv_ptr->image, depth_resized, cv::Size(768, 576), 0, 0, cv::INTER_NEAREST);
    // 获取当前时间戳并生成文件名
    timestampnow = rgb_msg->header.stamp;

    if (timestampnow - timestampold >= ros::Duration(0.5) && rotate_flag == 1) { // 如果时间差大于等于1秒
        std::stringstream rgb_filename, depth_filename, transform_filename;
        rgb_filename << std::setw(5) << std::setfill('0') << i;
        depth_filename << std::setw(5) << std::setfill('0') << i;
        transform_filename << std::setw(5) << std::setfill('0') << i;
        i++;

        std::string rgb_file = rgb_storage_path + rgb_filename.str() + ".jpg";
        std::string depth_file = depth_storage_path + depth_filename.str() + ".png";
        std::string transform_file = transform_storage_path + transform_filename.str() + ".txt";

        ROS_INFO("Saved RGB image: %s", rgb_file.c_str());
        ROS_INFO("Saved depth image: %s", depth_file.c_str());

        // 从 tf_msg1 中提取 map 与 livox_frame 之间的变换
        Eigen::Affine3d map_to_livox_frame;
        map_to_livox_frame.translation().x() = tf_msg1->pose.pose.position.x;
        map_to_livox_frame.translation().y() = tf_msg1->pose.pose.position.y;
        map_to_livox_frame.translation().z() = tf_msg1->pose.pose.position.z;
        Eigen::Quaterniond q(tf_msg1->pose.pose.orientation.w,
                             tf_msg1->pose.pose.orientation.x,
                             tf_msg1->pose.pose.orientation.y,
                             tf_msg1->pose.pose.orientation.z);
        map_to_livox_frame.linear() = q.toRotationMatrix();

        // 从 /tf_static 中提取 livox_frame 与 rgb_camera_link 之间的变换
        nav_msgs::Odometry tf_msg_static;
        tf_msg_static.pose.pose.position.x = 0.177144;
        tf_msg_static.pose.pose.position.y = -0.05637;
        tf_msg_static.pose.pose.position.z = 0.53938;
        tf_msg_static.pose.pose.orientation.w = 0.511142251457;
        tf_msg_static.pose.pose.orientation.x = -0.495863987461;
        tf_msg_static.pose.pose.orientation.y = 0.494289340018;
        tf_msg_static.pose.pose.orientation.z = -0.498528387416;

        Eigen::Affine3d livox_frame_to_rgb_camera_link;
        livox_frame_to_rgb_camera_link.translation().x() = tf_msg_static.pose.pose.position.x;
        livox_frame_to_rgb_camera_link.translation().y() = tf_msg_static.pose.pose.position.y;
        livox_frame_to_rgb_camera_link.translation().z() = tf_msg_static.pose.pose.position.z;
        Eigen::Quaterniond q_static(tf_msg_static.pose.pose.orientation.w,
                                    tf_msg_static.pose.pose.orientation.x,
                                    tf_msg_static.pose.pose.orientation.y,
                                    tf_msg_static.pose.pose.orientation.z);
                                     livox_frame_to_rgb_camera_link.linear() = q_static.toRotationMatrix();

        // 计算 rgb_camera_link 与 map 之间的变换
        Eigen::Affine3d rgb_camera_link_to_map = map_to_livox_frame * livox_frame_to_rgb_camera_link;

        // 存储 RGB 和深度图像
        cv::imwrite(rgb_file, rgb_resized);
        cv::imwrite(depth_file, depth_resized);

        // 保存变换信息到文件
        std::ofstream transform_file_stream(transform_file);
        transform_file_stream << rgb_camera_link_to_map.matrix() << std::endl;
        transform_file_stream.close();

        // 更新上次存储时间戳
        timestampold = timestampnow;
    }
            }
        }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "image_storage_example");
    ros::NodeHandle nh;

//删除与重建目录
    if (bf::exists(newobv_storage_path) && bf::is_directory(newobv_storage_path)) {
        for (bf::directory_iterator it(newobv_storage_path); it != bf::directory_iterator(); ++it) {
            bf::remove_all(*it);
        }
        std::cout << "Directory " << newobv_storage_path << " cleared successfully!" << std::endl;
    } else {
        std::cout << "Directory " << newobv_storage_path << " does not exist or is not a directory." << std::endl;
    }
  // 检查文件夹是否存在
        if (!bf::exists(rgb_storage_path)) {
            // 创建文件夹
            if (bf::create_directory(rgb_storage_path)) {
                std::cout << "Folder created successfully: " << rgb_storage_path << std::endl;
            } else {
                std::cerr << "Failed to create folder: " << rgb_storage_path << std::endl;
            }
        } else {
            std::cout << "Folder already exists: " << rgb_storage_path << std::endl;
        }

    // 检查文件夹是否存在
        if (!bf::exists(depth_storage_path)) {
            // 创建文件夹
            if (bf::create_directory(depth_storage_path)) {
                std::cout << "Folder created successfully: " << depth_storage_path << std::endl;
            } else {
                std::cerr << "Failed to create folder: " << depth_storage_path << std::endl;
            }
        } else {
            std::cout << "Folder already exists: " << depth_storage_path << std::endl;
        }

    // 检查文件夹是否存在
        if (!bf::exists(transform_storage_path)) {
            // 创建文件夹
            if (bf::create_directory(transform_storage_path)) {
                std::cout << "Folder created successfully: " << transform_storage_path << std::endl;
            } else {
                std::cerr << "Failed to create folder: " << transform_storage_path << std::endl;
            }
        } else {
            std::cout << "Folder already exists: " << transform_storage_path << std::endl;
        }

    std::ifstream file(filename.c_str());
    file.seekg(0, std::ios::end);
    lastSize =file.tellg() ;
    // printf("lastSize: %ld !!!!!!!!!!\n", static_cast<long>(lastSize));
    // Create subscribers
    message_filters::Subscriber<sensor_msgs::Image> depth_sub(nh, "/depth_to_rgb/image_raw", 5);
    message_filters::Subscriber<sensor_msgs::Image> rgb_sub(nh, "/rgb/image_raw", 5);
    message_filters::Subscriber<nav_msgs::Odometry> tf_sub1(nh, "/odom_tf", 40);
    // ros::Subscriber int_sub = nh.subscribe("/rotate_flag", 2, rotate_flag_Callback);

    // message_filters::Subscriber<std_msgs::Int32> flag_sub(nh, "/rotateflag", 5);
    // ROS_INFO("timestampold: %f", timestampold);

    // Create ApproximateTimeSynchronizer
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, nav_msgs::Odometry> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(15), depth_sub, rgb_sub, tf_sub1);

    // printf("lastSize: %ld !!!!!!!!!!\n", static_cast<long>(lastSize));
    // Register the callback function
    sync.registerCallback(boost::bind(&callback, _1, _2, _3));

    ros::spin();
    return 0;
}