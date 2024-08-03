# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d
import random

def read_bin_file(file_path):
    # 读取二进制点云数据
    point_cloud = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)  # 4列: x, y, z, intensity
    print(f"点云数据形状: {point_cloud.shape}")
    return point_cloud

def read_label_file(file_path):
    # 读取标签数据
    labels = np.fromfile(file_path, dtype=np.uint32)
    print(f"标签数据形状: {labels.shape}")
    print(f"唯一标签及计数: {np.unique(labels, return_counts=True)}")
    return labels

def generate_random_colors(num_colors):
    # 生成随机颜色
    return np.random.rand(num_colors, 3)

def visualize_point_cloud(points, labels=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    if labels is not None:
        if len(labels) != len(points):
            raise ValueError("标签数据长度与点云数据长度不匹配")
        
        # 为每个唯一标签生成随机颜色
        unique_labels = np.unique(labels)
        num_labels = len(unique_labels)
        colors = generate_random_colors(num_labels)
        
        # 创建颜色数组并初始化为黑色（未知标签）
        color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
        color_array = np.zeros((points.shape[0], 3))
        
        # 应用颜色到点云
        for label, color in color_map.items():
            mask = (labels == label)
            print(f"处理标签 {label}: 颜色 {color}, 数量 {np.sum(mask)}")
            if np.sum(mask) > 0:
                color_array[mask] = color
        
        pcd.colors = o3d.utility.Vector3dVector(color_array)
    
    o3d.visualization.draw_geometries([pcd])

# 文件路径
bin_file_path = '/home/admin123/test/0000.bin'
label_file_path = '/home/admin123/test/output.label'

# 读取数据
points = read_bin_file(bin_file_path)
labels = read_label_file(label_file_path)

# 可视化
visualize_point_cloud(points, labels)
