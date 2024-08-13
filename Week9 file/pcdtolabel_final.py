# -*- coding: utf-8 -*-
import struct
import os

def read_pcd_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Find the line where the data starts
    data_start_index = 0
    for i, line in enumerate(lines):
        if line.strip() == "DATA ascii":
            data_start_index = i + 1
            break

    # Read point cloud data
    points = []
    labels = []
    for line in lines[data_start_index:]:
        parts = line.strip().split()
        x = float(parts[0])
        y = float(parts[1])
        z = float(parts[2])
        label = int(parts[4])
        points.append([x, y, z])
        labels.append(label)

    return points, labels

def save_to_bin_file(points, file_path):
    with open(file_path, 'wb') as f:
        for point in points:
            f.write(struct.pack('fff', point[0], point[1], point[2]))

def save_to_label_file(labels, file_path):
    with open(file_path, 'w') as f:
        for label in labels:
            f.write("{0}\n".format(label))

def pcd_to_bin_and_label(pcd_file_path, output_bin_path, output_label_path):
    points, labels = read_pcd_file(pcd_file_path)
    save_to_bin_file(points, output_bin_path)
    save_to_label_file(labels, output_label_path)

def convert_directory(input_directory, output_directory):
    # Create the output directory if it does not exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for file_name in os.listdir(input_directory):
        if file_name.endswith('.pcd'):
            pcd_file_path = os.path.join(input_directory, file_name)
            output_bin_path = os.path.join(output_directory, file_name.replace('.pcd', '.bin'))
            output_label_path = os.path.join(output_directory, file_name.replace('.pcd', '.label'))
            pcd_to_bin_and_label(pcd_file_path, output_bin_path, output_label_path)
            print('PCD data from {} has been saved to {} and {}'.format(pcd_file_path, output_bin_path, output_label_path))

# 示例使用
input_directory = './realpcd'  # 输入PCD文件的目录
output_directory = './output'  # 输出二进制文件和标签文件的目录

convert_directory(input_directory, output_directory)
