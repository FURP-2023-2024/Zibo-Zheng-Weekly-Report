# -*- coding: utf-8 -*-
import rosbag
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import time

def read_point_cloud_from_bag(bag_file_path, topic_name):

    bag = rosbag.Bag(bag_file_path, 'r')
    point_clouds = []
    message_count = 0  # 计数器
    
    for topic, msg, t in bag.read_messages(topics=[topic_name]):
        
        # print("Reading message {} from topic: {}".format(message_count, topic))  # 调试输出
        # print("Message type: {}".format(type(msg)))  # 打印消息类型

        # if isinstance(msg, PointCloud2):
        # print("!!!!!")
        # print("Message header: {}".format(msg.header))  # 打印消息头
            
        if hasattr(msg.header, 'stamp'):
            # print("Timestamp in header: {}".format(msg.header.stamp))  # 打印时间戳字段
            timestamp = msg.header.stamp.to_sec()
            print("Converted timestamp: {}".format(timestamp))  # 打印转换后的时间戳
            print("id: {}".format(message_count)) 
            # points = []
            # try:
            #     for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            #         points.append([point[0], point[1], point[2]])
            # except Exception as e:
            #     print("Error reading points: {}".format(e))
            #     continue
            
            point_clouds.append((timestamp,message_count))
            # print("Read point cloud with timestamp: {}".format(timestamp))  # 调试输出
        else:
            print("No timestamp found in message header")
        message_count += 1
    bag.close()
    return point_clouds

# 文件路径和主题名称
bag_file_path = '20240602_1.bag'
topic_name = '/rslidar_points'

# 文件路径
file_path = 'optimized_odom_tum.txt'  # 替换为你的文件路径

# 超时时间设置为10秒
timeout_duration = 10

# 获取当前时间
start_time = time.time()
first_number=[]
# 打开并读取文件
tot=0
with open(file_path, 'r') as file:
    for line in file:
        # 获取当前时间
        current_time = time.time()
        if tot==980:
            break
        else:
            tot+=1
        # 检查是否超过了超时时间
        if current_time - start_time > timeout_duration:
            print("超过10秒未读入数据，退出循环")
            break
        
        # 去除行首尾的空白字符（包括换行符）
        stripped_line = line.strip()
        
        # 如果行为空，继续读取下一行
        if not stripped_line:
            continue
        
        # 使用空格分隔行中的内容
        parts = stripped_line.split()
        
        # 取第一个元素并尝试转换为浮点数（如果需要）
        if parts:
            try:
                first_number.append(float(parts[0]))
                # print(first_number)
            except ValueError:
                print("无法转换 '{}' 为浮点数".format(parts[0]))

        # 更新开始时间（可选，依赖于特定的应用逻辑）
        start_time = time.time()

# 读取点云数据和时间戳
point_clouds = read_point_cloud_from_bag(bag_file_path, topic_name)

# 打印每个点云帧的时间戳

print("start to find the closest id")
sum=0
for item in first_number:
    minn=1000000
    for point_cloud in point_clouds:
        timestamp, message_count =  point_cloud[0], point_cloud[1]
        # print(item,timestamp)
        if abs(item-timestamp)<minn:
            # print(11)
            # print("找到点云帧的时间戳: {}".format(timestamp))
            minn=abs(item-timestamp)
            id=message_count
    # print(id)
    sum+=1
    if sum==4:
        print(id)
        sum=0


# 打印读取到的点云帧数量
print("Total number of point cloud frames read: {}".format(len(point_clouds)))
