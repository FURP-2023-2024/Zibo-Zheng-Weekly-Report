# -*- coding: utf-8 -*-

import os
import rosbag
import sensor_msgs.point_cloud2 as pc2
import numpy as np

def bag_to_pcd(bag_file, pcd_folder):
    # 打开 .bag 文件
    bag = rosbag.Bag(bag_file)
    
    # 计数器，用于生成文件名
    count = 0
    
    for topic, msg, t in bag.read_messages(topics=['/rslidar_points']):
        # 提取点云数据
        points = pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z", "rgb"))
        
        # 将点云数据转换为 numpy 数组
        point_list = []
        for point in points:
            # 处理 RGB 信息
            if len(point) == 4:  # 说明有 RGB 信息
                r = (int(point[3]) >> 16) & 0xFF
                g = (int(point[3]) >> 8) & 0xFF
                b = int(point[3]) & 0xFF
                rgb = (r * 65536 + g * 256 + b) / (255.0 * 255.0 * 255.0)  # RGB 归一化到 [0, 1]
            else:
                rgb = 0.0  # 如果没有 RGB 信息，设置为默认值
            
            point_list.append([point[0], point[1], point[2], rgb])
        
        point_array = np.array(point_list, dtype=np.float32)
        
        # 创建 .pcd 文件
        pcd_file = os.path.join(pcd_folder, '{:04d}.pcd'.format(count))
        with open(pcd_file, 'w') as f:
            # 写入 .pcd 文件头部信息
            f.write("# .PCD v0.7 - Point Cloud Data file format\n")
            f.write("VERSION 0.7\n")
            f.write("FIELDS x y z rgb\n")
            f.write("SIZE 4 4 4 4\n")
            f.write("TYPE F F F F\n")
            f.write("COUNT 1 1 1 1\n")
            f.write("WIDTH {}\n".format(point_array.shape[0]))
            f.write("HEIGHT 1\n")
            f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
            f.write("POINTS {}\n".format(point_array.shape[0]))
            f.write("DATA ascii\n")
            
            # 写入点云数据
            for point in point_array:
                f.write("{} {} {} {}\n".format(point[0], point[1], point[2], point[3]))
        print(count)
        count += 1
    
    bag.close()

# 使用示例
bag_file = '/home/admin123/real_pc/20240602_1.bag'
pcd_folder = 'output_pcd_files'
if not os.path.exists(pcd_folder):
    os.makedirs(pcd_folder)
bag_to_pcd(bag_file, pcd_folder)
