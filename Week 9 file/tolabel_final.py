# -*- coding: utf-8 -*-
import struct
import os

def read_label_file(file_path):
    with open(file_path, 'r') as f:
        labels = [int(line.strip()) for line in f.readlines()]
    return labels

def save_labels_to_bin_file(labels, output_bin_path):
    with open(output_bin_path, 'wb') as f:
        for label in labels:
            f.write(struct.pack('i', label))

def label_to_bin(label_file_path, output_bin_path):
    labels = read_label_file(label_file_path)
    save_labels_to_bin_file(labels, output_bin_path)

def convert_label_directory(input_directory, output_directory):
    # Create the output directory if it does not exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for file_name in os.listdir(input_directory):
        if file_name.endswith('.label'):
            label_file_path = os.path.join(input_directory, file_name)
            output_bin_path = os.path.join(output_directory, file_name)
            label_to_bin(label_file_path, output_bin_path)
            print('Label data from {} has been saved to {}'.format(label_file_path, output_bin_path))

# 示例使用
input_directory = './output'  # 输入.label文件的目录
output_directory = './output_label'  # 输出二进制.label文件的目录

convert_label_directory(input_directory, output_directory)
