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
        if hasattr(msg.header, 'stamp'):
            timestamp = msg.header.stamp.to_sec()
            point_clouds.append((timestamp, message_count))
        message_count += 1
    bag.close()
    return point_clouds

# 文件路径和主题名称
bag_file_path = '20240602_1.bag'
topic_name = '/rslidar_points'

# 文件路径
file_path = 'optimized_odom_tum.txt'  # 替换为你的文件路径
output_file_path = 'matched_lines.txt'  # 输出文件路径

# 超时时间设置为10秒
timeout_duration = 10

# 获取当前时间
start_time = time.time()
first_number = []
lines = []
# 打开并读取文件
tot = 0
with open(file_path, 'r') as file:
    for line in file:
        current_time = time.time()
        if tot == 980:
            break
        else:
            tot += 1
        if current_time - start_time > timeout_duration:
            print("超过10秒未读入数据，退出循环")
            break
        
        stripped_line = line.strip()
        if not stripped_line:
            continue
        
        parts = stripped_line.split()
        if parts:
            try:
                first_number.append(float(parts[0]))
                lines.append(line)
            except ValueError:
                print("无法转换 '{}' 为浮点数".format(parts[0]))

        start_time = time.time()

# 读取点云数据和时间戳
point_clouds = read_point_cloud_from_bag(bag_file_path, topic_name)

# 打印每个点云帧的时间戳
print("start to find the closest id")

matched_lines = []
sum = 0
for item in first_number:
    minn = 1000000
    closest_line = None
    for point_cloud in point_clouds:
        timestamp, message_count = point_cloud[0], point_cloud[1]
        if abs(item - timestamp) < minn:
            minn = abs(item - timestamp)
            closest_line = lines[first_number.index(item)]
    sum += 1
    if sum == 4:
        if closest_line:
            print(closest_line.strip())
            matched_lines.append(closest_line.strip())
        sum = 0

# 写入新的文本文件
with open(output_file_path, 'w') as output_file:
    for matched_line in matched_lines:
        output_file.write(matched_line + '\n')

print("Total number of point cloud frames read: {}".format(len(point_clouds)))
print("Matched lines have been written to {}".format(output_file_path))
