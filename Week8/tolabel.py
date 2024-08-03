# -*- coding: utf-8 -*-
import struct

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

# 示例使用
label_file_path = 'output.label'  # 输入的文本格式的 .label 文件
output_bin_path = 'output_labels.label'  # 输出的二进制格式的 .label 文件

label_to_bin(label_file_path, output_bin_path)
print('Label data has been saved to {}'.format(output_bin_path))
