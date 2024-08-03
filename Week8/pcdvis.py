# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d
import random

def read_pcd_file(file_path):
    with open(file_path, 'r') as file:
        # 读取文件头
        header = ""
        while True:
            line = file.readline()
            if line.startswith("DATA ascii"):
                break
            header += line
        
        # 读取数据部分
        data = []
        for line in file:
            data.append(line.strip().split())

    # 转换为 numpy 数组
    data = np.array(data, dtype=np.float32)
    
    # 提取字段
    points = data[:, :3]  # x, y, z
    rgb = data[:, 3].astype(np.uint32)  # RGB颜色（目前不使用）
    labels = data[:, 4].astype(np.int32)  # 标签
    objects = data[:, 5].astype(np.int32)  # 物体（可选）

    return points, labels

def generate_random_colors(num_colors):
    # 为每个标签生成随机颜色
    return np.random.rand(num_colors, 3)

def visualize_point_cloud(points, labels):
    # 生成每个标签的随机颜色
    unique_labels = np.unique(labels)
    num_labels = len(unique_labels)
    colors = generate_random_colors(num_labels)
    
    # 创建颜色映射
    color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
    
    # 创建颜色数组并初始化为黑色（未知标签）
    color_array = np.zeros((points.shape[0], 3))
    
    # 应用颜色到点云
    for i, label in enumerate(labels):
        if label in color_map:
            color_array[i] = color_map[label]
    
    # 创建并可视化点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(color_array)
    
    o3d.visualization.draw_geometries([pcd])

# 文件路径
pcd_file_path = '/home/admin123/test/0000.pcd'

# 读取点云数据
points, labels = read_pcd_file(pcd_file_path)

# 可视化点云
visualize_point_cloud(points, labels)
