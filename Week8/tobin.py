import os
import struct
import rosbag
import sensor_msgs.point_cloud2 as pc2
import numpy as np

def bag_to_bin(bag_file, bin_folder):
    # 打开.bag文件
    bag = rosbag.Bag(bag_file)
    
    for topic, msg, t in bag.read_messages(topics=['/your_pointcloud_topic']):
        # 提取点云数据
        points = pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z", "intensity"))
        
        # 将点云数据转换为numpy数组
        point_list = []
        for point in points:
            point_list.append([point[0], point[1], point[2], point[3]])  # 这里假设点云包含强度信息
        point_array = np.array(point_list, dtype=np.float32)
        
        # 创建.bin文件
        bin_file = os.path.join(bin_folder, f'{t}.bin')
        with open(bin_file, 'wb') as f:
            # 使用struct库将数据写入.bin文件
            f.write(point_array.tobytes())
    
    bag.close()

# 使用示例
bag_file = 'your_bag_file.bag'
bin_folder = 'output_bin_files'
os.makedirs(bin_folder, exist_ok=True)
bag_to_bin(bag_file, bin_folder)
