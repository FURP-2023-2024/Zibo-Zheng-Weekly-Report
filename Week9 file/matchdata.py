# -*- coding: utf-8 -*-
import math

def read_xyz_from_file(file_path):
    xyz_coords = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 4:
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                xyz_coords.append((x, y, z))
    return xyz_coords

def calculate_distance(coord1, coord2):
    return math.sqrt((coord1[0] - coord2[0]) ** 2 + 
                     (coord1[1] - coord2[1]) ** 2 + 
                     (coord1[2] - coord2[2]) ** 2)

def process_distances(xyz_coords, output_file_path):
    with open(output_file_path, 'w') as output_file:
        for i in range(len(xyz_coords)):
            for j in range(i + 1, len(xyz_coords)):
                distance = calculate_distance(xyz_coords[i], xyz_coords[j])
                if distance < 2:
                    output_file.write("{} {} 1\n".format(i+1, j+1))
                elif distance > 30:
                    output_file.write("{} {} 0\n".format(i+1, j+1))

# 文件路径
input_file_path = 'matched_lines.txt'
output_file_path = 'output_distances.txt'

# 读取xyz坐标
xyz_coords = read_xyz_from_file(input_file_path)

# 处理并写入距离结果
process_distances(xyz_coords, output_file_path)

print("Distance processing complete. Results saved to {}".format(output_file_path))
