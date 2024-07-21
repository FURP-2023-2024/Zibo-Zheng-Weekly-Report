# -*- coding: UTF-8 -*-
import copy
import json
import os
import pickle
import gzip
import argparse

import matplotlib
import numpy as np
import pandas as pd
import open3d as o3d
import torch
import torch.nn.functional as F
from pathlib import Path
# import open_clip
import clip

# sbert模型
from sentence_transformers import SentenceTransformer, util

import distinctipy

from conceptgraph.slam.slam_classes import MapObjectList
from conceptgraph.slam.utils import filter_objects, merge_objects

import time
import threading
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, default=None)
    parser.add_argument("--class_path", type=str, default=None)
    parser.add_argument("--class_colors_file", type=str, default=None)
    parser.add_argument("--rgb_pcd_path", type=str, default=None)
    parser.add_argument("--edge_file", type=str, default=None)
    parser.add_argument("--poses_path", type=str, default=None)
    
    parser.add_argument("--no_clip", action="store_true", 
                        help="If set, the CLIP model will not init for fast debugging.")
    
    # To inspect the results of merge_overlap_objects
    # This is mainly to quickly try out different thresholds
    parser.add_argument("--merge_overlap_thresh", type=float, default=-1)
    parser.add_argument("--merge_visual_sim_thresh", type=float, default=-1)
    parser.add_argument("--merge_text_sim_thresh", type=float, default=-1)
    parser.add_argument("--obj_min_points", type=int, default=0)
    parser.add_argument("--obj_min_detections", type=int, default=0)
    
    return parser

def load_result(result_path):
    with gzip.open(result_path, "rb") as f:
        results = pickle.load(f)
    
    if isinstance(results, dict):  #is this
        # print("results are dict")
        objects = MapObjectList()
        objects.load_serializable(results["objects"])
        
        if results['bg_objects'] is None:
            bg_objects = None
        else:
            bg_objects = MapObjectList()
            bg_objects.load_serializable(results["bg_objects"])

        class_colors = results['class_colors']   #
        # print(len(class_colors),class_colors)
    elif isinstance(results, list):
        # print("results are list")
        objects = MapObjectList()
        objects.load_serializable(results)

        bg_objects = None
        class_colors = distinctipy.get_colors(len(objects), pastel_factor=0.5)  # 生成一组视觉上相异的颜色
        class_colors = {str(i): c for i, c in enumerate(class_colors)}
    else:
        raise ValueError("Unknown results type: ", type(results))
    
        
    return objects, bg_objects, class_colors

def T_normal_axis(normals, axis): #计算normals转到axis的4*4的T  ;第一步先计算normals的主normal   axis：[0, 0, 1]
    # #聚类法：取法向量
    # num_clusters = 2  # Number of clusters
    # kmeans = KMeans(n_clusters=num_clusters)
    # kmeans.fit(normals)
    # cluster_labels = kmeans.labels_
    # cluster_centers = kmeans.cluster_centers_
    # largest_cluster_label = np.argmax(np.bincount(cluster_labels))
    # major_normal = cluster_centers[largest_cluster_label]
    # if abs(cluster_centers[0][1]) > abs(cluster_centers[1][1]) :
    #     major_normal = cluster_centers[0]
    # else:
    #     major_normal = cluster_centers[1]
    
    # print(cluster_centers, major_normal)
    #特征向量法
    covariance_matrix = np.cov(normals, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    major_normal = eigenvectors[:, np.argmax(eigenvalues)]
    major_normal = -major_normal
    
    # Compute T matrix to align the normal vector with the z-axis
    angle = np.arccos(np.dot(major_normal, axis) / (np.linalg.norm(major_normal) * np.linalg.norm(axis)))
    # print("Angle with z-axis:", np.degrees(angle), "degrees")
    rot_axis = np.cross(major_normal, axis)
    rot_axis /= np.linalg.norm(rot_axis)
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c
    x, y, z = rot_axis
    T =  np.array([
        [t * x**2 + c, t * x * y - s * z, t * x * z + s * y, 0],
        [t * x * y + s * z, t * y**2 + c, t * y * z - s * x, 0],
        [t * x * z - s * y, t * y * z + s * x, t * z**2 + c, 0],
        [0, 0, 0, 1]
    ])
    # print(T)
    return T

#xyz投影到xy
def xyz2xy(list_pointcloud, text_name):   #list_pointcloud
    points = np.asarray(list_pointcloud)     #np.array
    # np.savetxt(text_name, points[:,0:2])
    # plt.figure(figsize=(8, 6))
    # plt.scatter(points[:, 0], points[:, 1], color='blue')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('all_points x-y points')
    # plt.grid(True)
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.show()
    return points[:,0:2]

def read_class_txt2list(filename):
    with open(filename, 'r') as file:
    # 读取文件内容并去除首尾的空白字符
        content = file.read().strip()
        # 将字符串转换为列表，使用eval函数来解析字符串中的列表
        text_list = eval(content)
    # print(text_list)
    return text_list

def draw_grid(grid_color, grid_xy, min_x, max_x, min_y, max_y, scatter_data, class_colors, object_classes_id):
    x_len, y_len = grid_color.shape
    color_id = np.unique(grid_color)

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    # 绘制每个单元格
    for i in range(x_len):
        for j in range(y_len):
            x, y = grid_xy[i][j]
            color_id = grid_color[i][j]
            rect = plt.Rectangle((x, y), 1, 1, facecolor=(min(0.07*color_id, 1), min(0.05*color_id, 1), min(0.03*color_id, 1)))
            ax.add_patch(rect)
    # 绘制散点scatter_data = [((0, 0), 1),)]
    scatter_coords = [coord for coord, color in scatter_data]
    scatter_colors = [color for coord, color in scatter_data]
    # for coord, color_id in zip(scatter_coords, scatter_colors):
    for i in range(len(scatter_coords)):
        x, y = scatter_coords[i]
        obj_class = object_classes_id[i]
        #TODO
        # if obj_class > 80:
        #     obj_class =-1
        plt.scatter(x, y, color=class_colors[str(obj_class)])
    # 设置网格范围
    ax.set_xlim([min_x, max_x])
    ax.set_ylim([min_y, max_y])
    
    # 显示网格
    # plt.grid(True, color='black', linewidth=1)
    plt.show()

def x_y_histogram(all_points, threshold=300):
    all_points = np.array(all_points)
    # 提取 x 和 y 坐标
    x_coords = all_points[:, 0]
    y_coords = all_points[:, 1]

    # 定义网格的边界和分辨率
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    grid_resolution = 0.1

    # 计算每个网格单元的点数
    x_bins = np.arange(x_min, x_max, grid_resolution)
    y_bins = np.arange(y_min, y_max, grid_resolution)
    hist, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=[x_bins, y_bins])

    # 标记墙体的网格单元
    wall_mask = hist > threshold

    # 可视化结果
    # plt.figure(figsize=(10, 8))
    # plt.title('Wall Detection')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.imshow(hist.T, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='hot', aspect='auto')
    # plt.colorbar(label='Number of Points')

    # # 绘制墙体
    # wall_x, wall_y = np.where(wall_mask)
    # wall_x = xedges[wall_x]
    # wall_y = yedges[wall_y]
    # plt.scatter(wall_x, wall_y, color='blue', label='Wall Points', s=1)

    # plt.legend()
    # plt.show()

def get_file_lines_len(file_path):
    """ 获取文件内容的哈希值 """
    with open(file_path, 'r') as file:
        lines = file.readlines()
        return len(lines)

def main(args): 
    # *加载参数
    result_path = args.result_path
    # class_path = args.class_path
    #!sg22habitat要用poses_path:注意在里面更改
    # poses_path = args.poses_path 

    rgb_pcd_path = args.rgb_pcd_path
    # class_colors_path = args.class_colors_file
    class_colors = None 
    # class_text_list = read_class_txt2list(class_path)
    assert not (result_path is None and rgb_pcd_path is None), \
        "Either result_path or rgb_pcd_path must be provided."
    # SBERT文本编码器
    sbert_model = SentenceTransformer('/home/admin123/robustnav/weights/all-MiniLM-L6-v2')
    print("import sbert!")
    clip_model, preprocess = clip.load("ViT-B/32", device="cuda")
    print("import clip!")
    #full pcd中有objects
    objects, bg_objects, class_colors = load_result(result_path)
    # if class_colors_path is not None:    
    #     with open(Path(args.class_colors_file), "r") as f:
    #         class_colors = json.load(f)
    cmap = matplotlib.colormaps.get_cmap("turbo")   # 颜色映射，将数据传给cmap后，会返回相应颜色的RGB值，在可视化中表示这些数据值
    if bg_objects is not None:
        indices_bg = np.arange(len(objects), len(objects) + len(bg_objects))
        objects.extend(bg_objects)
    print("11")
    from scipy.spatial.transform import Rotation as R
    # from sg22habitat import read_yaml_file, cvt_pose_vec2tf, cvt_tf2pose_vec, habitat2sg, sg2habitat, \
    #     base_pos2grid_id_2d, from_habitat_tf2_full2dmap_pos
    # print("import sg22habitat")
    

    # * 获取object的主要class_id
    # *Sub-sample the point cloud for better interactive experience
    # Get the color for each object when colored by their class
    object_classes_id = []
    for i in range(len(objects)):
        pcd = objects[i]['pcd']
        pcd = pcd.voxel_down_sample(0.05)
        objects[i]['pcd'] = pcd
        obj = objects[i]
        #一个obj是很多点，每个点的class可能不一样，所以要找这个obj的主要的class 类别
        obj_classes = np.asarray(obj['class_id'])   #obj['class_id']  是该pbj的每个点的class_id
        # print(obj_classes)
        # Get the most common class for this object as the class
        values, counts = np.unique(obj_classes, return_counts=True) # values为唯一值，counts为唯一值的数量
        obj_class = values[np.argmax(counts)]   # np.argmax返回数组中的最大值索引
        object_classes_id.append(obj_class)
        # print(class_text_list[obj_class])
        # print(len(object_classes_id))        floor:class_id:33?
    pcds = copy.deepcopy(objects.get_values("pcd"))
    bboxes = copy.deepcopy(objects.get_values("bbox"))
    # print("process object pcds and class_ids.")

    # o3d.visualization.draw_geometries(pcds)

    # #*---------应该不用做了，habitat转过来就是纠正好的---------#
    # # * 纠正物体点云的z轴
    # ceiling_num = 0
    # for i in range(len(class_text_list)):
    #     if class_text_list[i] == "floor":    #ceiling floor
    #         ceiling_num = i
    # floor_index = [index for index, value in enumerate(object_classes_id) if value == ceiling_num][0]  # class=33对应 floor
    # pcds[floor_index].estimate_normals()
    # # Access the normal information
    # normals_floor = np.asarray(pcds[floor_index].normals)
    # T_z = T_normal_axis(normals_floor, [0,0,1]) 
    # #使pcds的floor垂直于z轴；
    # for i in range(len(pcds)):
    #     pcds[i].transform(T_z)
    # #*---------应该不用做了，habitat转过来就是纠正好的---------#

    #* 计算最大和最小的z值:方式直方图
    # import matplotlib.pyplot as plt
    from scipy.signal import find_peaks
    z_values_all = []
    pcds = copy.deepcopy(objects.get_values("pcd"))
    for point_cloud in pcds:
        z_values = np.asarray(point_cloud.points)[:, 2]
        z_values_all.extend(z_values)
    hist, bins = np.histogram(z_values_all, bins=50)
    threshold = np.max(hist) * 0.3  # Adjust the threshold as needed
    peaks = np.where(hist > threshold)[0]
    peak_z_values = bins[peaks]
    max_z = np.max(peak_z_values)
    min_z = np.min(peak_z_values)
    # print(max_z, min_z)

    #! 找门
    text_class = 'table'
    #*定义caption  利用sbert特征在地图中找到他们
    # List of load-bearing objects
    from conceptgraph.utils.general_utils import to_tensor
    # Encode the features for all words in chengzai_list
    # item_ft = clip_model.encode_text(clip.tokenize(text_class).to("cuda"))
    item_ft = sbert_model.encode(text_class, convert_to_tensor=True, device="cuda")
    item_ft = item_ft / item_ft.norm(dim=-1, keepdim=True)
    threshold = 0.6  # Define your similarity threshold here
    candidate_door_id = []
    max_thre = 0
    door_indices = []
    for i, obj in enumerate(objects):
        # print(obj["class_id"], obj["class_name"])
        objects_text_ft = obj['text_ft']
        captions = obj['caption']
        objects_text_ft_tensor = to_tensor(objects_text_ft).to("cuda")
        text_sim = F.cosine_similarity(objects_text_ft_tensor, item_ft.unsqueeze(0), dim=-1)
        # Extract the similarity value from tensor
        text_sim_value = text_sim.item()
        # print(captions)
        num_door_in_caps = 0
        for caption in captions:
            if text_class in caption:
                num_door_in_caps +=1 
        if num_door_in_caps * 2  >  len(captions):
            door_indices.append(i)
        # if text_sim_value > threshold:
        #     candidate_door_id.append([i,text_sim_value])

    #* 找出一定高度区间的点云
    # 1. 取高处的墙体点云，分出房间轮廓  所以：假设采集的数据中墙的采集很充分
    # 2. 取地面和天花板之间的点云，分出障碍物轮廓
    #todo: min_z 不是0
    floor_height =-0.8
    room_height = max_z - floor_height
    plane_heights = 7 * room_height / 10 + floor_height  
    print("max_z,plane_heights,min_z: ",max_z,plane_heights,min_z)
    wall_points_by_height3d = []
    obstacle_points_by_height3d = []
    door_points = []
    all_points = []
    centers=[]
    from aobao_area import alpha_shape
    for i in range(len(pcds)):
        points = pcds[i].points
        
        points = np.asarray(points)
        
        #! debug 对于gibson_adrian_7，Y要求小于0.8m
        # selected_points = points[points[:, 1] < 0.9]
        selected_points = points
        if len(selected_points) ==0:
            continue
        # selected_points = points
        
        all_points.extend(selected_points)
        center = np.mean(points, axis=0)
        centers.append(center[0:3])

        # if i in door_indices:  #门相关的点全部记录下来
        #     door_points_2d =  xyz2xy(selected_points, "all_points.txt")
        #     # 计算 Alpha Shape
        #     alpha = 5  # 控制参数
        #     concave_hull, edge_points = alpha_shape(door_points_2d, alpha)
            
        #     # 计算面积
        #     area = concave_hull.area
        #     print(f"The area of the concave hull is: {area}")
        #     if area < 0.3:
        #         for point in selected_points:
        #             door_points.append(point)

        for point in selected_points:
            # 1. 取高处的墙体点云，分出房间轮廓  所以：假设采集的数据中墙的采集很充分
            if abs(point[2] - plane_heights) < 0.15:  #! 阈值，待调整
                wall_points_by_height3d.append(point)
            # 2. 取地面和天花板之间的点云，分出障碍物轮廓
            if (point[2] - floor_height > 0.2) and (max_z - point[2] > 0.3):  #这个阈值生成障碍物mask还可以
                obstacle_points_by_height3d.append(point)
    
    # x_y_histogram(wall_points_by_height3d)
    #! debug: 对于gibson_adrian_7，Y要求小于0.8m
    #!显示点云的二维投影
    # wall_points_by_height = xyz2xy(wall_points_by_height3d,"wall_points.txt")
    all_points_array = xyz2xy(all_points, "all_points.txt")
    obstacle_points_array = xyz2xy(obstacle_points_by_height3d, "all_points.txt")
    
    
    # plt.figure(figsize=(10, 10))
    # # plt.scatter(all_points_array[:, 0], all_points_array[:, 1], color='blue', label='All Points')
    # plt.scatter(wall_points_by_height[:, 0], wall_points_by_height[:, 1], color='red', label='Wall Points')
    # plt.scatter(obstacle_points_array[:, 0], obstacle_points_array[:, 1], color='green', label='Wall Points')
    # # plt.scatter(door_points_array[:, 0], door_points_array[:, 1], color='green', label='Wall Points')
    # plt.legend()
    # plt.xlabel('X-axis Label')
    # plt.ylabel('Y-axis Label')
    # plt.title('Scatter Plot of Wall Points and All Points')
    # plt.show()


    # # * 物体分配房间号  2D分出多个房间，判断每个物体在哪个房间

    # from test_room_segment_1 import room_main
    # room_object_belong_list = room_main(wall_points_by_height, all_points_array ,obstacle_points_by_height3d_array , door_points_array, np.array(centers))

    room_object_belong_list = []
    # print(np.unique(dilated_matrix))
    for object_center in centers:
        # #!
        # if point[1] > 0.8:
        #     continue
        # object_center_where = point_in_ornot_region(object_center, dilated_matrix, grid_x_y, min_x, max_x, min_y, max_y, grid_resolution)
        room_object_belong_list.append([(object_center[0], object_center[1], object_center[2]), 1])
    # print(room_object_belong_list)
    # draw_grid(dilated_matrix, grid_x_y, min_x, max_x, min_y, max_y, room_object_belong_list, class_colors, object_classes_id)



    #*分出承载级物体
    #定义尺寸列表，记录每个物体尺寸；背景类物体为-1
    size_list = []  
    
    #选承载类candidate: 要求承载类物体的长宽高都大于0.4m，点云与地面接触，中心点高度< 房间高度/2
    # todo:要统计物体点云的密度
    chengzai_candidate_ids = []
    
    # for i, obj in enumerate(objects):
    for i in range(len(objects)):
        if i in indices_bg:  #排除背景类物体
            size_list.append(-1)
            continue
        pcd = pcds[i]
        points = np.asarray(pcd.points)
        center = centers[i]
        x_max = points[:, 0].max()
        x_min = points[:, 0].min()
        y_max = points[:, 1].max()
        y_min = points[:, 1].min()
        z_max = points[:, 2].max()
        z_min = points[:, 2].min()
        size_list.append((x_max-x_min)*(y_max-y_min)*(z_max-z_min))
        print(z_max, z_min, y_max, y_min, x_max, x_min, floor_height)
        if abs(z_min-floor_height)<0.1:  #接触地面
            if (z_max - z_min) <1.5:   #排除墙壁
                if x_max-x_min >0.4 and y_max-y_min >0.4 and center[2]-floor_height >0.2:  
                    # if x_max-x_min < 2 or  y_max-y_min < 2:
                    chengzai_candidate_ids.append(i)
                    print("chengzai_candidate_ids: ",chengzai_candidate_ids[-1])
                    print("the room id they belongs to: ",room_object_belong_list[i])

    #todo: chengzai_candidate_ids 再经过sbert和clip feat的“承载”功能检验和筛选
    
    
    
    #*定义scene graph
    ## 房间-->物体
    room_dict = {}
    obj_id=0
    for obj_center, room_id in room_object_belong_list:
        if obj_id >= indices_bg[0]:  #房间中 只保存非背景类物体
            break
        if room_id not in room_dict:
            room_dict[room_id] = []  # Start with []
        room_dict[room_id].append(obj_id)
        obj_id +=1
    # rooms_to_objs = list(room_dict.values())
    room_ids = list(room_dict.keys())
    rooms_to_objs = room_dict   #room_dict={'room1': [obj1,obj2], 'room2': [obj1,obj2],}
    print("rooms_to_objs: ",rooms_to_objs)
    
    ## 承载级物体-->承载的小物体
    chengzai_to_objs =  {}        #chengzai_to_objs = {"chengzai1":[room_id, obj1,obj2], "chengzai2":[room_id, obj1,obj2]}
    chengzai_to_objs_probability = {} # chengzai_to_objs_probability = {"chengzai1":[room_id, obj1_prob,obj2_prob], }
    for chengzai_id in chengzai_candidate_ids:
        chengzai_pcd = pcds[chengzai_id]
        chengzai_points = np.asarray(chengzai_pcd.points)
        chengzai_center = centers[chengzai_id]
        chengzai_x_max = chengzai_points[:, 0].max()
        chengzai_x_min = chengzai_points[:, 0].min()
        chengzai_y_max = chengzai_points[:, 1].max()
        chengzai_y_min = chengzai_points[:, 1].min()
        chengzai_z_max = chengzai_points[:, 2].max()
        chengzai_z_min = chengzai_points[:, 2].min()   

        room_id = room_object_belong_list[chengzai_id][1]  #chengzai_id所在rooms
        small_objs_in_room = rooms_to_objs[room_id] 
        if chengzai_id not in chengzai_to_objs:
            chengzai_to_objs[chengzai_id] = [room_id]  # Start with the room_id in the list
            chengzai_to_objs_probability[chengzai_id] = [room_id]
        for small_obj in small_objs_in_room:
            small_obj_pcd = pcds[small_obj]
            small_obj_points = np.asarray(small_obj_pcd.points)
            small_obj_center = centers[small_obj]
            small_obj_x_max = small_obj_points[:, 0].max()
            small_obj_x_min = small_obj_points[:, 0].min()
            small_obj_y_max = small_obj_points[:, 1].max()
            small_obj_y_min = small_obj_points[:, 1].min()
            small_obj_z_max = small_obj_points[:, 2].max()
            small_obj_z_min = small_obj_points[:, 2].min()
            # 判断 small_obj 是否在 chengzai_obj 上面
            #todo：得到的被承载小物体有些是墙，是否能根据sbert或clip特征筛除他们
            #! 是否可以以二维上判断承载级和小物体的iou为主，其他指标辅助
            is_above = (
    small_obj_center[2] >= chengzai_center[2] and 
    small_obj_z_min - chengzai_z_max <= 0.05 and
    max_z - small_obj_z_max>0.2 and # small_obj 的底部高于 chengzai_obj 的顶部
    # 有个枕头的点云尺寸的一个维度>1m
    abs(small_obj_x_max - small_obj_x_min)<=1.2 and abs(small_obj_y_max - small_obj_y_min)<=1.2 and #小物体的尺寸不大
    abs(small_obj_x_max - small_obj_x_min)>0.05 and abs(small_obj_y_max - small_obj_y_min)>0.05 and  #尺寸
    (small_obj_center[0] < chengzai_x_max+0.02) and 
    (small_obj_center[0] > chengzai_x_min -0.02) and  # small_obj 在 x 方向上与 chengzai_obj 有重叠
    (small_obj_center[1] < chengzai_y_max+0.02) and 
    (small_obj_center[1] > chengzai_y_min-0.02)  # small_obj 在 y 方向上与 chengzai_obj 有重叠
            )
            if is_above and small_obj != chengzai_id:
                chengzai_to_objs[chengzai_id].append(small_obj)
                chengzai_to_objs_probability[chengzai_id].append(1)
    print(chengzai_to_objs)
    # print(chengzai_to_objs_probability)
    


    print(chengzai_to_objs)
    #* 实例级物体查找: 目前的导航模式是多样的，房间-->承载--->小物体  
    object_description = ['bed in the room 2.']
    #接入大模型，判断：房间号，（承载级物体），目标物体

    
    Task_done_flag = 1 #导航任务是否完成的flag  没有导航任务也算已完成

    detection_flag_file_path = '/home/admin123/robustnav/concept-graphs/conceptgraph/flag_files/front_detection_finish.txt'
    initial_detection_flag_file_length = get_file_lines_len(detection_flag_file_path)
    multi_targets_list = []     #[[room_id, chengzai_id, obj_id], ]
    while True:
    # for i in range(1):
        ## "a pillow on the bed in room 2."
        #!如果当前没有新任务，就等待输入下一个任务
        if Task_done_flag == 1:
            user_input = input("请输入文本（输入 'exit' 退出）：")  
            if user_input.lower() == 'exit':
                print("程序结束。")
                break
            
            # #接入大模型，判断：房间号，（承载级物体），目标物体
            # #todo:要保留形容词描述，比如small
            # nav_prompt_instance = "Extract three elements from the input navigation instructions: room number (digits), carrier-level object name, and target small object. The output format example is: 6 table banana. Note that there are two special cases: 1. If the carrier-level object name is not provided, the output format example is: 6 none banana; 2. If the target small object name is not provided, the output format example is: 6 table none. We generally define carrier-level objects as: table, bed, sofa, etc. The navigation instructions are: "    
            nav_prompt_instance = "Extract three elements from the given navigation instructions: room number (numeric),\
                             carrier object (which may include adjectives and other descriptors and should be extracted), \
                                and target small object (which may include adjectives and other descriptors and should be extracted). \
                            The output format example is directly (without any other words): 6 table banana or 6 table banana. \
                            Note, there are two special cases: 1. If no carrier object name is given, the output format example is: 6 none banana; \
                                2. If no target small object name is given, the output format example is: 6 table none; \
                                    We generally define carrier objects as 【table, bed, sofa】 etc.\
                                        The given navigation instruction is: "
            

            #! openai 0.28.0
            import openai
            def call_gpt4_api_image(prompt, image=None):

                openai.api_key = 'sk-nSLftdDUHwnDxwXu244e1290B3944f6c8e3111066337730d'
                
                openai.api_base = 'https://api.xiaoai.plus/v1'
                
                # "gpt-4-vision-preview"
                response = openai.ChatCompletion.create(
                    model="gpt-4-vision-preview",
                    messages=[
                    {
                        "role": "user",
                            "content": 
                            [
                                {"type": "text", "text": prompt}
                            ]
                    }
                    ],
                    max_tokens=3000,
                )
                    
                print(response.choices[0].message['content'])
                return response.choices[0].message['content']
            
            #* 判断导航类型：指定物体的导航 / 需求类型导航
            nav_target_description = user_input
            nav_type_prompt = "If the input is something like a need, \
                            such as [I'm thirsty] or [I want to read], please output 1. If the input is a specific item, \
                            such as [bed in the room 1] or [book on the table], please response 2. "
            nav_type = call_gpt4_api_image(nav_type_prompt+"input: "+nav_target_description)
            nav_type = int(nav_type)
            #指定物体的导航
            if int(nav_type) == 2:
                task_room_id, task_chengzai_obj_class, task_small_obj_class = 2,'bed', 'mug'
                room2chengzai2small_ = call_gpt4_api_image(nav_prompt_instance+nav_target_description) 
            
                task_room_id, task_chengzai_obj_class, task_small_obj_class = room2chengzai2small_.split(" ")
                
                print("指定物体的导航: task_room_id, task_chengzai_obj_class, task_small_obj_class: ", task_room_id, task_chengzai_obj_class, task_small_obj_class)
                task_room_id = int(task_room_id)
            
            #需求类型导航
            if int(nav_type) == 1:
                # room2chengzai2small_ = call_gpt4_api_image(nav_prompt_instance+nav_target_description) 
            
                # task_room_id, task_chengzai_obj_class, task_small_obj_class = room2chengzai2small_.split(" ")
                task_room_id, task_chengzai_obj_class, task_small_obj_class = -1,'none', nav_target_description
                print("需求类型导航: task_room_id, task_chengzai_obj_class, task_small_obj_class: ", task_room_id, task_chengzai_obj_class, task_small_obj_class)
                task_room_id = int(task_room_id)
            
    

            #1. 指定了承载级物体,是否找被承载的小物体待定
            #生成chengzai_obj_class特征
            if task_chengzai_obj_class !="none":
                cehngzai_item_ft_clip = clip_model.encode_text(clip.tokenize(task_chengzai_obj_class).to("cuda"))
                cehngzai_item_ft_sbert = sbert_model.encode(task_chengzai_obj_class, convert_to_tensor=True, device="cuda")
                cehngzai_item_ft_clip = cehngzai_item_ft_clip / cehngzai_item_ft_clip.norm(dim=-1, keepdim=True)
                cehngzai_item_ft_sbert = cehngzai_item_ft_sbert / cehngzai_item_ft_sbert.norm(dim=-1, keepdim=True)
                cehngzai_item_ft_clip = cehngzai_item_ft_clip.to("cuda")
                cehngzai_item_ft_sbert = cehngzai_item_ft_sbert.to("cuda")

                max_chengzai_sim = 0; selected_chengzai_id = 0
                #定位房间和承载级物体
                for chengzai_id in list(chengzai_to_objs.keys()):
                    #先选择房间
                    if chengzai_to_objs[chengzai_id][0] !=task_room_id:
                        continue
                    #承载级物体匹配
                    objects_clip_ft = objects[chengzai_id]['clip_ft'].to("cuda")
                    chengzai_objects_text_ft = objects[chengzai_id]['text_ft'].to("cuda")
                    chengzai_objects_text_ft_tensor = to_tensor(chengzai_objects_text_ft).to("cuda")

                    text_sim = F.cosine_similarity(chengzai_objects_text_ft_tensor, cehngzai_item_ft_sbert.unsqueeze(0), dim=-1)
                    # Extract the similarity value from tensor
                    text_sim_value = text_sim.item()
                    if text_sim_value > max_chengzai_sim:
                        max_chengzai_sim = text_sim_value
                        selected_chengzai_id = chengzai_id
                print("max_chengzai_sim: ",max_chengzai_sim, " chengzxai_id: ",selected_chengzai_id)
                #2. 只找承载级物体
                if task_small_obj_class == "none":
                    point_pose = np.array([centers[selected_chengzai_id][0],  centers[selected_chengzai_id][1],  1.5,  0.5, -0.5, 0.5, -0.5])
                    selected_obj_id = selected_chengzai_id
                    print("the sg position of carrier object is: ",centers[selected_chengzai_id][0]," ",  centers[selected_chengzai_id][1])

                #3. 找承载级物体上的小物体
                else:
                    if len(chengzai_to_objs[selected_chengzai_id]) - 1 == 0:
                        print(task_chengzai_obj_class+" in room "+room_id+ "carry no objects.")
                    small_task_ft_clip = clip_model.encode_text(clip.tokenize(task_small_obj_class).to("cuda"))
                    small_task_ft_sbert = sbert_model.encode(task_small_obj_class, convert_to_tensor=True, device="cuda")
                    small_task_ft_clip = small_task_ft_clip / small_task_ft_clip.norm(dim=-1, keepdim=True)
                    small_task_ft_sbert = small_task_ft_sbert / small_task_ft_sbert.norm(dim=-1, keepdim=True)
                    small_task_ft_clip = small_task_ft_clip.to("cuda")
                    small_task_ft_sbert = small_task_ft_sbert.to("cuda")

                    max_sim = 0; selected_obj_id = 0;  task_obj_id = 0
                    print("selected_chengzai_id: ", selected_chengzai_id, "chengzai object contains these objs: ", chengzai_to_objs[selected_chengzai_id])
                    for obj_id in chengzai_to_objs[selected_chengzai_id][1:]: #找chengzai_id上的小东西
                        #小物体匹配
                        objects_clip_ft = objects[obj_id]['clip_ft'].to("cuda")
                        objects_text_ft = objects[obj_id]['text_ft'].to("cuda")
                        objects_text_ft_tensor = to_tensor(objects_text_ft).to("cuda")

                        text_sim = F.cosine_similarity(objects_text_ft_tensor, small_task_ft_sbert.unsqueeze(0), dim=-1)
                        # Extract the similarity value from tensor
                        text_sim_value = text_sim.item()
                        if text_sim_value > max_sim:
                            max_sim = text_sim_value
                            selected_obj_id = obj_id
                    point_pose = np.array([centers[selected_obj_id][0],  centers[selected_obj_id][1],  1.5,  0.5, -0.5, 0.5, -0.5])

                    #todo: 当max_sim太低,不认为找到了目标小物体
                    print("max_sim: ", max_sim)
                    print("the sg position of small object is: ",selected_obj_id, " ",centers[selected_obj_id][0]," ",  centers[selected_obj_id][1])

                    task_obj_id = selected_obj_id
            #4. 没指定承载级物体,只是找被承载的小物体
            else:
                small_task_ft_clip = clip_model.encode_text(clip.tokenize(task_small_obj_class).to("cuda"))
                small_task_ft_sbert = sbert_model.encode(task_small_obj_class, convert_to_tensor=True, device="cuda")
                small_task_ft_clip = small_task_ft_clip / small_task_ft_clip.norm(dim=-1, keepdim=True)
                small_task_ft_sbert = small_task_ft_sbert / small_task_ft_sbert.norm(dim=-1, keepdim=True)
                small_task_ft_clip = small_task_ft_clip.to("cuda")
                small_task_ft_sbert = small_task_ft_sbert.to("cuda")

                max_sim = 0; selected_obj_id = 0

                obj_lists = []
                #需求类型导航
                if int(nav_type) == 1:
                    obj_lists = range(len(objects))
                #指定物体的导航
                elif int(nav_type) == 2:
                    obj_lists = rooms_to_objs[room_id]

                for obj_id in obj_lists: #指定房间内找物体
                    #小物体匹配
                    objects_clip_ft = objects[obj_id]['clip_ft'].to("cuda")
                    objects_text_ft = objects[obj_id]['text_ft'].to("cuda")
                    objects_text_ft_tensor = to_tensor(objects_text_ft).to("cuda")

                    text_sim = F.cosine_similarity(objects_text_ft_tensor, small_task_ft_sbert.unsqueeze(0), dim=-1)
                    # Extract the similarity value from tensor
                    text_sim_value = text_sim.item()
                    if text_sim_value > max_sim:
                        max_sim = text_sim_value
                        selected_obj_id = obj_id
                print("max_chengzai_sim: ", max_sim)
                point_pose = np.array([centers[selected_obj_id][0],  centers[selected_obj_id][1],  1.5,  0.5, -0.5, 0.5, -0.5])
                print("the target object is: ",centers[selected_obj_id])

        #!如果当前已有任务，打印出任务的名称:
        else:
        # task_room_id, task_chengzai_obj_class, task_small_obj_class = 6, "bed", "none"
            print("task_room_id, task_chengzai_obj_class, task_small_obj_class: ", task_room_id, selected_chengzai_id, selected_obj_id)
            point_pose = np.array([centers[selected_obj_id][0],  centers[selected_obj_id][1],  1.5,  0.5, -0.5, 0.5, -0.5])
        #写入当前目标位置，供仿真器读取
        with open('/home/admin123/catkin_ws/src/nav_pkg/src/your_file.txt', 'a') as file:
            file.write('\n' + str(centers[selected_obj_id][0]) + " " + str(centers[selected_obj_id][1]-1) + " " + str(centers[selected_obj_id][2])+" 0")

        robot_pos_id = selected_obj_id    #默认机器人会到目标位置,记录id

        print("Waiting for robot's navigation.")

        #! 场景图更新：这一步应该到达终点再做，并判断small 是否还存在
        ##################################################################################################################
        #*:判断obs是否更新
        while True:
        
            current_detection_flag_file_length = get_file_lines_len(detection_flag_file_path)

            if current_detection_flag_file_length != initial_detection_flag_file_length:
                print("Robot finishes nav and detection.")
                break
        
        print("Start updating the scene graph!")
        initial_detection_flag_file_length = current_detection_flag_file_length
        
        import glob
        new_observation_data_dir = "/data/dyn/igibson_dataset/saved_data_rotate_1/detect_out/"
        # file_paths = glob.glob(os.path.join(new_observation_data_dir, '*.pkl.gz'))
        file_paths = sorted(glob.glob(os.path.join(new_observation_data_dir, '*.pkl.gz')))
    
        data_list = []
        size_thre = 0.5   #size1/size2 >0.5
        distance_thre = 0.7  #2m以内
        clip_feat_thre = 0.7  #  视觉特征相似
        sbert_feat_thre = 0.7  #  语义特征相似
        depth_thre = 4       #观察中的物体到相机的距离阈值，不考虑过远的物体
        
        #读每一帧新obs
        # for i in range(0,len(file_paths),2):  #2帧读取一次
        new_objs_added = []    #增加到所有承载级的新小物体   [[room_id,chengzai_id,new_obj_id, caption]]
        old_objs_disappeared = []  #所有承载级上消失的小物体 
        old_objs_observed = []     #如果old object在本轮更新被观察到了，那么本轮不再删除它； 如果有新加入的物体，那么本轮不再删除它

        #! 机器人导航过程中，认为环境不变，所以只可能去除地图中id小于old_objects_num-1的物体
        old_objects_num = len(objects) 

        for i in range(0,len(file_paths),1):  
        # for i in range(0,18,1):

            selected_chengzai_ids = []  #观察和地图匹配的承载级物体  [[chengzai_id, j],..]   

            file_path = new_observation_data_dir + str(i) + '.pkl.gz'
            print(file_path)
            with gzip.open(file_path, 'rb') as f:
                new_obs_datas = pickle.load(f)
                #先找匹配的承载级，即观察中出现了几个场景图中的承载级物体
                for chengzai_id in list(chengzai_to_objs.keys()):
                    #! debug
                    # chengzai_id =26
                    pcd_chengzai = pcds[chengzai_id]
                    points_chengzai = np.array(pcd_chengzai.points)
                    x_max_chengzai = points_chengzai[:, 0].max()
                    x_min_chengzai = points_chengzai[:, 0].min()
                    y_max_chengzai = points_chengzai[:, 1].max()
                    y_min_chengzai = points_chengzai[:, 1].min()
                    z_max_chengzai = points_chengzai[:, 2].max()
                    z_min_chengzai = points_chengzai[:, 2].min()
                    

                    objects_clip_ft = objects[chengzai_id]['clip_ft'].to("cuda")
                    objects_sbert_ft = objects[chengzai_id]['text_ft'].to("cuda")
                    
                    clip_ft_tensor_chengzai_objects = to_tensor(objects_clip_ft).to("cuda")
                    sbert_ft_tensor_chengzai_objects = to_tensor(objects_sbert_ft).to("cuda")
                    center_chengzai = centers[chengzai_id]
                    size_chengzai  = size_list[chengzai_id]

                    # print("x y z chengzai: ",chengzai_id, center_chengzai, size_chengzai, x_max_chengzai, x_min_chengzai, y_max_chengzai,\
                    #        y_min_chengzai, z_max_chengzai, z_min_chengzai)
                    # print(chengzai_id, center_chengzai, size_chengzai)
                    for j in range(len(new_obs_datas["mask"])):  # 假设所有列表的长度相同
                        mask = new_obs_datas["mask"][j]
                        bbox = new_obs_datas["xyxy"][j]
                        caption = new_obs_datas["caption"][j]
                        # print("111111")
                        # 识别为 房间/场景 等笼统意义的物体不考虑
                        if ("room" in caption)  or ("scene" in  caption):
                            continue
                        # print("caption", caption)
                        
                        clip_feat_obj = new_obs_datas["image_feats"][j]
                        text_feat = new_obs_datas["text_feats"][j]
                        confidence = new_obs_datas["confidence"][j]
                        posed_pcd_point = new_obs_datas["posed_pcd_points_image"][j]
                        depth_mean = new_obs_datas["depth_mean_image"][j]
                        # print("depth_mean: ", depth_mean)
                        if depth_mean >=depth_thre:
                            continue
                        # pcd = o3d.geometry.PointCloud()
                        # pcd.points = o3d.utility.Vector3dVector(posed_pcd_point[::3])
                        # pcd.voxel_down_sample(0.1)
                        # pcds.append(pcd)

                        x_max = posed_pcd_point[:, 0].max()
                        x_min = posed_pcd_point[:, 0].min()
                        y_max = posed_pcd_point[:, 1].max()
                        y_min = posed_pcd_point[:, 1].min()
                        z_max = posed_pcd_point[:, 2].max()
                        z_min = posed_pcd_point[:, 2].min()
                        
                        size_obj = (x_max-x_min)*(y_max-y_min)*(z_max-z_min)
                        center_obj = np.mean(posed_pcd_point, axis=0)
                        #! debug
                        # if caption == "a banana":
                        #     print('pcds and objects['pcd'] num differs')
                        #     pcd = o3d.geometry.PointCloud()
                        #     pcd.points = o3d.utility.Vector3dVector(posed_pcd_point)
                        #     # pcd.voxel_down_sample(0.1)
                        #     pcds.append(pcd)
                        # #     print("center_obj: ", center_obj)
                        #     # print("x y z ", x_max, x_min, y_max, y_min, z_max, z_min)
                        clip_feat_obj = new_obs_datas["image_feats"][j]
                        clip_feat_obj_tensor = torch.from_numpy(clip_feat_obj).to("cuda")
                        sbert_feat_obj = new_obs_datas["text_feats"][j]
                        sbert_feat_obj_tensor = torch.from_numpy(sbert_feat_obj).to("cuda")

                        clips_sim = F.cosine_similarity(clip_ft_tensor_chengzai_objects, clip_feat_obj_tensor.unsqueeze(0), dim=-1)
                        clips_sim_value = clips_sim.item()
                        sbert_sim = F.cosine_similarity(sbert_ft_tensor_chengzai_objects, sbert_feat_obj_tensor.unsqueeze(0), dim=-1)
                        sbert_sim_value = sbert_sim.item()

                        size_rate = min(size_obj/size_chengzai, size_chengzai/size_obj)
                        center_distance = np.linalg.norm(center_obj - center_chengzai)
                        # print("new_obj caption, sbert_sim_value, clips_sim, size_rate, center_distance: ",caption, sbert_sim_value, clips_sim_value, size_rate, center_distance)
                        if sbert_sim_value >=sbert_feat_thre and clips_sim_value >=clip_feat_thre and size_rate >= size_thre and center_distance <=distance_thre:
                            selected_chengzai_ids.append([chengzai_id, j])  
                print("selected_chengzai_ids: ", selected_chengzai_ids)
                #匹配承载物体上的小物体，做增，减，移动的检查
                print("匹配承载物体上的小物体，做增，减，移动的检查")
                for chengzai_id, new_obs_chengzai_id in selected_chengzai_ids:
                    pcd_chengzai = pcds[chengzai_id]
                    points_chengzai = np.array(pcd_chengzai.points)
                    
                    points_new_obv_chengzai = new_obs_datas["posed_pcd_points_image"][new_obs_chengzai_id]
                    pcd_new_obv_chengzai = o3d.geometry.PointCloud()
                    pcd_new_obv_chengzai.points = o3d.utility.Vector3dVector(points_new_obv_chengzai)

                    objects_clip_ft = objects[chengzai_id]['clip_ft'].to("cuda")
                    objects_sbert_ft = objects[chengzai_id]['text_ft'].to("cuda")
                    
                    clip_ft_tensor_chengzai_objects = to_tensor(objects_clip_ft).to("cuda")
                    sbert_ft_tensor_chengzai_objects = to_tensor(objects_sbert_ft).to("cuda")
                    center_chengzai = centers[chengzai_id]
                    print("x y z chengzai: ",chengzai_id, center_chengzai, size_chengzai)
                    # print("x y z chengzai: ",chengzai_id, center_chengzai, size_chengzai, x_max_chengzai, x_min_chengzai, y_max_chengzai,\
                    #        y_min_chengzai, z_max_chengzai, z_min_chengzai)
                
                    new_small_obj_lis = [] #记录该承载级上最新观察到的small_obj
                    for j in range(len(new_obs_datas["mask"])):  
                        
                        caption = new_obs_datas["caption"][j]
                        # 识别为 房间/场景 等笼统意义的物体不考虑
                        if ("room" in caption)  or ("scene" in  caption):
                            continue
                        # print("caption", caption)
                        
                        clip_feat_obj = new_obs_datas["image_feats"][j]
                        posed_pcd_point = new_obs_datas["posed_pcd_points_image"][j]

                        depth_mean = new_obs_datas["depth_mean_image"][j]
                        if depth_mean >=depth_thre:
                            continue

                        x_max = posed_pcd_point[:, 0].max()
                        x_min = posed_pcd_point[:, 0].min()
                        y_max = posed_pcd_point[:, 1].max()
                        y_min = posed_pcd_point[:, 1].min()
                        z_max = posed_pcd_point[:, 2].max()
                        z_min = posed_pcd_point[:, 2].min()
                        
                        size_obj = (x_max-x_min)*(y_max-y_min)*(z_max-z_min)
                        center_obj = np.mean(posed_pcd_point, axis=0)
                        if is_above(points_chengzai, center_chengzai, posed_pcd_point, center_obj, max_z):
                            # print("This new obj is_above: caption, center_obj, size_obj: ",caption, center_obj, size_obj, x_max, x_min, y_max,\
                        #    y_min, z_max, z_min)
                            new_small_obj_lis.append(j)

                    room_id = chengzai_to_objs[chengzai_id][0]
                    # 1. 承载物体上原本没有small objs, 此时new_small_obj_lis直接加进来
                    if not chengzai_to_objs[chengzai_id][1:]:
                        print("old_small_objs on chengzai_id: ", chengzai_id, " is empty.")
                        for new_small_obj_id in new_small_obj_lis:
                            
                            #!!!!!一定要deepcopy
                            new_object = copy.deepcopy(objects[-1])
                            objects.append(new_object)
                            old_objs_observed.append(len(objects)-1)
                            caption = new_obs_datas["caption"][new_small_obj_id]

                            posed_pcd_point = new_obs_datas["posed_pcd_points_image"][new_small_obj_id]
                            pcd = o3d.geometry.PointCloud()
                            pcd.points = o3d.utility.Vector3dVector(posed_pcd_point)
                            pcd.colors = o3d.utility.Vector3dVector(
                                            np.tile(
                                                [1,0,0],
                                                (len(pcd.points), 1)
                                            )
                                            )
                            objects[-1]['pcd'] = pcd
                            pcds.append(pcd)
                            center = np.mean(pcd.points, axis=0)
                            
                            objects[-1]['clip_ft'] =   torch.from_numpy(new_obs_datas["image_feats"][new_small_obj_id]).to("cuda")
                            objects[-1]['text_ft'] = torch.from_numpy(new_obs_datas["text_feats"][new_small_obj_id]).to("cuda")
                            # objects[-1]['caption'] = caption
                            centers.append(np.array([center[0], center[1], center[2]]))
                            # print("centers[len(objects)]: ",len(objects)-1," ", centers[len(objects)-1])

                            #130: room 2 的bed 130 
                            print("Append: ",len(objects)-1, new_obs_datas["caption"][new_small_obj_id], new_small_obj_id)
                            chengzai_to_objs[chengzai_id].append(len(objects)-1)
                            rooms_to_objs[room_id].append(len(objects)-1)
                            new_objs_added.append([room_id, chengzai_id, len(objects)-1, caption])
                        print("After updated, chengzai_id ",chengzai_id,"  contains: ",chengzai_to_objs[chengzai_id])  
                    # 2. 承载物体上原本有small objs
                    else:
                        #找已有的small_objs和new_objs中相同的
                        to_remove = []   #去掉匹配上的new_small_obj_id, [new_small_obj_id,]  
                        for old_small_objs in chengzai_to_objs[chengzai_id][1:]:
                            # print("old_small_objs on chengzai_id: ", chengzai_id, " is not empty.")
                            # print("pcds len is :" , len(pcds), " old_small_objs id is : ", old_small_objs)
                            pcd_old_small_objs = pcds[old_small_objs]
                            points_old_small_objs = np.array(pcd_old_small_objs.points)
                            
                            
                            objects_clip_ft = objects[old_small_objs]['clip_ft']
                            objects_clip_ft = objects_clip_ft.to("cuda")
                            # objects_clip_ft = torch.from_numpy(objects_clip_ft).to("cuda")
                            # print(old_small_objs, objects_clip_ft[0:20])
                            objects_sbert_ft = objects[old_small_objs]['text_ft']
                            # objects_sbert_ft = torch.from_numpy(objects_sbert_ft).to("cuda")
                            objects_sbert_ft = objects_sbert_ft.to("cuda")
                            clip_ft_tensor_old_small_objs_objects = to_tensor(objects_clip_ft).to("cuda")
                            sbert_ft_tensor_old_small_objs_objects = to_tensor(objects_sbert_ft).to("cuda")

                            
                            is_still_here_flag = False

                            #! new obs中的该承载级若只观察了一部分，那么承载级上的old small objs不一定出镜了  对不出境的small obj，不考虑；
                            # pcd_new_small = o3d.geometry.PointCloud()
                            # pcd_new_small.points = o3d.utility.Vector3dVector(posed_pcd_point)
                            min_dist_c = min_dist_2_pcds(pcd_new_obv_chengzai, pcd_old_small_objs)
                            if min_dist_c > 0.1:   #如果new obs中承载级和 sg中该承载级的small obj距离大
                                print("min_dist_c > 0.1:   #如果new obs中承载级和 sg中该承载级的small obj距离大: ",old_small_objs)
                                continue

                            for new_small_obj_id in new_small_obj_lis:
                                posed_new_small_obj = new_obs_datas["posed_pcd_points_image"][new_small_obj_id]
                                depth_mean = new_obs_datas["depth_mean_image"][new_small_obj_id]
                                # print("depth_mean: ", depth_mean)
                                #! 离相机远的不考虑
                                if depth_mean >=depth_thre:
                                    continue

                                clip_feat_obj = new_obs_datas["image_feats"][new_small_obj_id]
                                clip_feat_new_small_obj_tensor = torch.from_numpy(clip_feat_obj).to("cuda")
                                sbert_feat_obj = new_obs_datas["text_feats"][new_small_obj_id]
                                sbert_feat_new_small_obj_tensor = torch.from_numpy(sbert_feat_obj).to("cuda")
                                
                                new_obs_caption = new_obs_datas["caption"][new_small_obj_id]
                                # print(caption)


                                #2.1 原先的被承载small obj匹配上new_objs（且位置不动）  去掉匹配上的new_small_obj_id
                                if is_still_here(points_old_small_objs, clip_ft_tensor_old_small_objs_objects, sbert_ft_tensor_old_small_objs_objects,
                                                 posed_new_small_obj,   clip_feat_new_small_obj_tensor,        sbert_feat_new_small_obj_tensor, new_obs_caption, old_small_objs):
                                    old_objs_observed.append(old_small_objs)
                                    to_remove.append(new_small_obj_id)
                                    is_still_here_flag = True
                                    print("This small object is still here: ", new_obs_datas["caption"][new_small_obj_id], new_small_obj_id)
                            

                            #2.2 地图上的被承载small obj已不在原位 去除以 更新sg, 并在old_objs_disappeared记录
                            if not is_still_here_flag and (old_small_objs not in old_objs_observed) :
                                print("old_small_objs to remove: ", old_small_objs)
                                chengzai_to_objs[chengzai_id].remove(old_small_objs)
                                objects[old_small_objs]['pcd'].colors = o3d.utility.Vector3dVector(np.tile([0, 0, 1], (len(pcds[old_small_objs].points), 1)))
                                pcds[old_small_objs].colors = o3d.utility.Vector3dVector(np.tile([0, 0, 1], (len(pcds[old_small_objs].points), 1)))
                                if old_small_objs in rooms_to_objs[room_id]:
                                    rooms_to_objs[room_id].remove(old_small_objs)
                                old_objs_disappeared.append([room_id, chengzai_id, old_small_objs])
                        
                        #去掉原本也存在的新观察中的objs
                        to_remove = list(set(to_remove))
                        for to_remove_new_small_obj_id in to_remove:
                            new_small_obj_lis.remove(to_remove_new_small_obj_id)    
                        print("new objs to_remove: ", to_remove)   
                        print("new objs left: ", new_small_obj_lis)   
                        #2.3 更新sg 在sg地图新增剩余的small物体, 在new_objs_added记录
                        for new_small_obj_id in new_small_obj_lis:
                            old_objs_observed.append(new_small_obj_id)
                            depth_mean = new_obs_datas["depth_mean_image"][new_small_obj_id]
                            caption = new_obs_datas["caption"][new_small_obj_id]
                            # print("depth_mean: ", depth_mean)
                            if depth_mean >=depth_thre:
                                continue
                            new_object = copy.deepcopy(objects[-1])
                            objects.append(new_object)

                            posed_pcd_point = new_obs_datas["posed_pcd_points_image"][new_small_obj_id]
                            pcd = o3d.geometry.PointCloud()
                            pcd.points = o3d.utility.Vector3dVector(posed_pcd_point)
                            pcd.colors = o3d.utility.Vector3dVector(
                                            np.tile(
                                                [1,0,0],
                                                (len(pcd.points), 1)
                                            )
                                            )
                            objects[-1]['pcd'] = pcd
                            pcds.append(pcd)
                            center = np.mean(pcd.points, axis=0)

                            objects[-1]['clip_ft'] =   torch.from_numpy(new_obs_datas["image_feats"][new_small_obj_id]).to("cuda")
                            objects[-1]['text_ft'] = torch.from_numpy(new_obs_datas["text_feats"][new_small_obj_id]).to("cuda")

                            centers.append(np.array([center[0], center[1], center[2]]))
                            # print("centers[len(objects)]: ",len(objects)-1," ", centers[len(objects)-1])

 
                            chengzai_to_objs[chengzai_id].append(len(objects)-1)
                            rooms_to_objs[room_id].append(len(objects)-1)
                            print("Append: ",len(objects)-1, new_obs_datas["caption"][new_small_obj_id], new_small_obj_id)
                            new_objs_added.append([room_id, chengzai_id, len(objects)-1, caption])

                        print("After updated, chengzai_id ",chengzai_id,"  contains: ",chengzai_to_objs[chengzai_id]) 
            # print(rooms_to_objs)
            # print(chengzai_to_objs)
            print("old_objs_observed: ",old_objs_observed)
        ##################################################################################################################

        #多个可能目标的导航先后排序
        def sort_multi_targets_list(robot_pos_id, multi_targets_list):
            distances = []
            for target in multi_targets_list:
                target_obj_id = target[2]
                robot_pos = np.array([centers[robot_pos_id][0],  centers[robot_pos_id][1]])
                target_obj_pos = np.array([centers[target_obj_id][0],  centers[target_obj_id][1]])
                distance = euclidean_distance(np.array(robot_pos), np.array(target_obj_pos))
                distances.append((distance, target))
            
            # 按距离排序
            distances.sort(key=lambda x: x[0])
            
            # 返回排序后的目标点
            sorted_targets = [target for _, target in distances]
            return sorted_targets

        #todo:先人工判断，后面换gpt
        #第一次导航
        if multi_targets_list == []:
            if task_chengzai_obj_class !="none" and task_small_obj_class !="none":
                #!要不要判断chengzai_to_objs[selected_chengzai_id][1:]为空
                #要找的物体已经不在这了   #todo:改成chengzai_to_objs[selected_chengzai_id]中每个small和目标的相似度对比一下
                if selected_obj_id not in chengzai_to_objs[selected_chengzai_id][1:]:  
                    
                    find_flag = 0
                    # prompt
                    # call_gpt4_api_image


                    for i, new_obj_caption_added in  enumerate(new_objs_added):
                        
                        # print("task_small_obj_class and new_obj_caption_added: ",task_small_obj_class ," ",new_obj_caption_added)
                        #新发现的物体中可能存在要找的目标，还有戏，开始多/单目标导航到可能的地方  
                        new_obj_sbert_ft = objects[new_obj_caption_added[2]]['text_ft'].to("cuda")
                        new_obj_sbert_ft_tensor = to_tensor(new_obj_sbert_ft).to("cuda")
                        text_sim = F.cosine_similarity(new_obj_sbert_ft_tensor, small_task_ft_sbert.unsqueeze(0), dim=-1)
                        print(text_sim, new_obj_caption_added[3])
                        if text_sim >0.6:

                        # if task_small_obj_class in new_obj_caption_added[3]:
                            multi_targets_list.append(new_obj_caption_added)

                            objects_task_obj_clip_ft = objects[task_obj_id]['clip_ft'].to("cuda")
                            clip_ft_tensor_objects_task__obj = to_tensor(objects_task_obj_clip_ft).to("cuda")

                            object_to_arrived_clip_ft = objects[new_obj_caption_added[2]]['clip_ft'].to("cuda")
                            clip_ft_tensor_object_arrived_clip_ft = to_tensor(object_to_arrived_clip_ft).to("cuda")
                            clips_sim = F.cosine_similarity(clip_ft_tensor_object_arrived_clip_ft, clip_ft_tensor_objects_task__obj.unsqueeze(0), dim=-1)
                            print("the clip simi is : ",clips_sim)

                            find_flag = 1
                            Task_done_flag = False
                            # selected_obj_id = new_objs_added[i][2]
                            # selected_chengzai_id = new_objs_added[i][1]
                            # task_room_id = new_objs_added[i][0]
                    print(multi_targets_list)
                    if multi_targets_list !=[]:
                        #todo:多个可能目标的 距离\clip特征加权 先后排序  robot_pos
                        robot_pos_id = robot_pos_id
                        multi_targets_list = sort_multi_targets_list(robot_pos_id, multi_targets_list)
                        #! 先导航到第一个candidate
                        selected_obj_id = multi_targets_list[0][2]
                        selected_chengzai_id = multi_targets_list[0][1]
                        task_room_id = multi_targets_list[0][0]

                        for target_candidate in multi_targets_list:
                            print("Small object is moved, but may be found here: ",\
                                target_candidate[0], " ",target_candidate[1], " ",target_candidate[2])
                        print("Move to the first new target： ",\
                              task_room_id, " ", selected_chengzai_id, " ", selected_obj_id)
                    
                    # 找不到了，下一个目标吧：或者直接上探索式
                    if find_flag == 0:
                        Task_done_flag =True
                        print("Task failed! Next Task:")
                #要找的物体还在这
                elif selected_obj_id in chengzai_to_objs[selected_chengzai_id][1:]:
                    Task_done_flag =True
                    print("Task done! Next Task:")
            else:
                Task_done_flag =True
                print("Task done! Next Task:")
        # 第一次导航的目标不在，且已经导航到至少一个可能的candidate
        else:
            #! 如果clip相似度大于0.8，认为找到正确的，结束   text_ft   clip_ft
            objects_task_obj_clip_ft = objects[task_obj_id]['text_ft'].to("cuda")   
            clip_ft_tensor_objects_task__obj = to_tensor(objects_task_obj_clip_ft).to("cuda")

            object_arrived_clip_ft = objects[selected_obj_id]['text_ft'].to("cuda")
            clip_ft_tensor_object_arrived_clip_ft = to_tensor(object_arrived_clip_ft).to("cuda")
            clips_sim = F.cosine_similarity(clip_ft_tensor_object_arrived_clip_ft, clip_ft_tensor_objects_task__obj.unsqueeze(0), dim=-1)
            if clips_sim >0.8:
                Task_done_flag =True
                multi_targets_list = []
                print("Task done! Find the moved object here. Next Task:")
            # 要接着找下一个candidate；如果没有下一个candidate，那任务失败咯
            else:
                print("The arrived candidate object mismatches the moved object. \
                      Try next candidate object.")
                robot_pos_id = robot_pos_id
                del multi_targets_list[0]
                # 没有下一个candidate，那任务失败咯
                if multi_targets_list == []:
                    print("Task failed! ALL candidate objects mismatches the moved object. Next Task:")
                    Task_done_flag =True
                #接着找下一个candidate
                else:
                    multi_targets_list = sort_multi_targets_list(robot_pos_id, multi_targets_list)
                    selected_obj_id = multi_targets_list[0][2]
                    selected_chengzai_id = multi_targets_list[0][1]
                    task_room_id = multi_targets_list[0][0]
                    print("The arrived candidate object mismatches the moved object. \
                      Try next candidate object.")


    #*显示点云
    vis = o3d.visualization.VisualizerWithKeyCallback()
    # vis = o3d.visualization.Visualizer()
    if result_path is not None:
        vis.create_window(window_name=f'Open3D - {os.path.basename(result_path)}', width=1280, height=720)
    else:
        vis.create_window(window_name=f'Open3D', width=1280, height=720)
   
    for i in range(len(pcds)):
        # if i in chengzai_objects_id[0:2]:
        #     random_color = np.random.rand(3)  # Generates 3 random numbers between 0 and 1
        #     pcds[i].paint_uniform_color(random_color)        
        vis.add_geometry(pcds[i]) 

    #*显示max_z和min_z的面点云
    # x_range = np.arange(-10, 0, 1)
    # y_range = np.arange(0, 10, 1)
    # points = []
    # for x in x_range:
    #     for y in y_range:
    #         points.append([x, y, max_z])
    #         points.append([x, y, min_z])
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # vis.add_geometry(pcd)

    # *红色显示一定高度的室内点云
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(obstacle_points_by_height3d)
    # pcd.paint_uniform_color([1, 0, 0])  # Assign different colors for different heights
    # vis.add_geometry(pcd)

    # * 显示坐标系
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    vis.add_geometry(coord_frame)
        
    main.show_bg_pcd = True
    def toggle_bg_pcd(vis): #切换
        if bg_objects is None:
            print("No background objects found.")
            return
        
        for idx in indices_bg:
            if main.show_bg_pcd:
                vis.remove_geometry(pcds[idx], reset_bounding_box=False)
                vis.remove_geometry(bboxes[idx], reset_bounding_box=False)
            else:
                vis.add_geometry(pcds[idx], reset_bounding_box=False)
                vis.add_geometry(bboxes[idx], reset_bounding_box=False)
        
        main.show_bg_pcd = not main.show_bg_pcd
        
    def color_by_class(vis):  #根据objects的主要class为pcd上色
        # print("class_colors: ", len(class_colors))
        # print(class_colors[str(object_classes_id[0])])
        for i in range(len(objects)):
            pcd = pcds[i]
            obj_class = object_classes_id[i]
            if obj_class>len(class_colors):
                print(obj_class)
            if obj_class > 80:
                obj_class =-1
            pcd.colors = o3d.utility.Vector3dVector(
                np.tile(
                    class_colors[str(obj_class)],
                    (len(pcd.points), 1)
                )
            )

        for pcd in pcds:
            vis.update_geometry(pcd)

    def color_by_room_class(vis):  #根据objects所属的room为pcd上色
        # print("class_colors: ", len(class_colors))
        # print(class_colors[str(object_classes_id[0])])
        room_class_ids = np.unique([item[1] for item in room_object_belong_list])
        print(room_class_ids)
        colormap = plt.cm.get_cmap("tab20", len(room_class_ids))
        for i in range(len(objects)):
            pcd = pcds[i]
            room_class = room_object_belong_list[i][1]
            index = np.where(room_class_ids == room_class)[0][0]
            # print("room_class, color is :", room_class, np.array([colormap(index)[0],colormap(index)[1],colormap(index)[2]]))
            pcd.colors = o3d.utility.Vector3dVector(
                np.tile(
                    np.array([colormap(index)[0],colormap(index)[1],colormap(index)[2]]),
                    (len(pcd.points), 1)
                )
            )
        for pcd in pcds:
            vis.update_geometry(pcd)
    
    def color_by_name(vis):
        text_query = input("Enter your name query: ")
        work_name_num = work_names.index(text_query)
        for i in objects_belong_work_names[work_name_num]:
            pcd = pcds[i]
            pcd.colors = o3d.utility.Vector3dVector(
            np.tile(
                [1,0,0],
                (len(pcd.points), 1)
            )
            )
            vis.update_geometry(pcd)

    def color_by_chengzai_candidate(vis):   #color出承载级物体
        colormap = plt.cm.get_cmap("hsv", len(chengzai_candidate_ids))
        i = 0
        for id in chengzai_candidate_ids:
            pcd = pcds[id]
            pcd.colors = o3d.utility.Vector3dVector(
            np.tile(
                np.array([colormap(i)[0],colormap(i)[1],colormap(i)[2]]),
                (len(pcd.points), 1)
            )
            )
            i = i+1
            vis.update_geometry(pcd)

    def color_by_chengzai_small_obj(vis):
        colormap = plt.cm.get_cmap("hsv", len(chengzai_candidate_ids))
        i = 0
        for chengzai_candidate_id in chengzai_candidate_ids:
            small_objs = chengzai_to_objs[chengzai_candidate_id][1:]
            for small_obj_id in small_objs:
                pcd = pcds[small_obj_id]
                pcd.colors = o3d.utility.Vector3dVector(
                np.tile(
                    np.array([colormap(i)[0],colormap(i)[1],colormap(i)[2]]),
                    (len(pcd.points), 1)
                )
                )
                i = i+1
                vis.update_geometry(pcd)

    def color_by_rgb(vis):   #根据pcd每个点的rgb颜色为每个点上色
        for i in range(len(pcds)):
            pcd = pcds[i]
            pcd.colors = objects[i]['pcd'].colors
        
        for pcd in pcds:
            vis.update_geometry(pcd)
            
    def color_by_instance(vis):
        instance_colors = cmap(np.linspace(0, 1, len(pcds)))
        for i in range(len(pcds)):
            pcd = pcds[i]
            if i%2 == 0:
                color_i = i
            else:
                color_i = len(pcds) -i
            pcd.colors = o3d.utility.Vector3dVector(
                np.tile(
                    instance_colors[color_i, :3],
                    (len(pcd.points), 1)
                )
            )
            
        for pcd in pcds:
            vis.update_geometry(pcd)
    
    def color_by_sbert_sim(vis):
        text_query = input("Enter your sbert query: ")
        # text_queries = [text_query]
        
        # caption_ft = sbert_model.encode(text_query, convert_to_tensor=True, device="cuda")
        # caption_ft = caption_ft / caption_ft.norm(dim=-1, keepdim=True)
        # caption_ft = caption_ft.detach().cpu().numpy()
        
        caption_ft = sbert_model.encode(text_query, convert_to_tensor=True, device="cuda")
        caption_ft = caption_ft / caption_ft.norm(dim=-1, keepdim=True)
        caption_ft = caption_ft.to("cuda")

        # similarities = objects.compute_similarities(text_query_ft)
        objects_text_fts = objects.get_stacked_values_torch("text_ft")
        objects_text_fts = objects_text_fts.to("cuda")
        similarities = F.cosine_similarity(
            caption_ft.unsqueeze(0), objects_text_fts, dim=-1
        )
        max_value = similarities.max()
        min_value = similarities.min()
        normalized_similarities = (similarities - min_value) / (max_value - min_value)
        probs = F.softmax(similarities, dim=0)      # 将每一列的相似性值映射到[0,1]区间内，得到概率分布
        max_prob_idx = torch.argmax(probs)      # 返回最大值的元素索引
        similarity_colors = cmap(normalized_similarities.detach().cpu().numpy())[..., :3]   # 将相似性值映射为颜色
        
        # 更新点云对象的颜色属性，以反映每个对象的相似性
        for i in range(len(objects)):
            pcd = pcds[i]
            map_colors = np.asarray(pcd.colors) # 先保存一下原有的颜色信息，但好像也没啥用
            pcd.colors = o3d.utility.Vector3dVector(        # 将numpy数组转换为open3d中的颜色向量格式
                np.tile(
                    [
                        similarity_colors[i, 0].item(),     # 第i行第一个元素，即红色通道的值，并得到标量值
                        similarity_colors[i, 1].item(),
                        similarity_colors[i, 2].item()
                    ], 
                    (len(pcd.points), 1)    # 使用np.tile函数将similarity_colors[i] 沿着第一个轴（行轴）复制成指定的行数，保证为每个点都分配相同的颜色
                )
            )

        for pcd in pcds:
            vis.update_geometry(pcd)

    def color_by_clip_sim(vis):
        # if args.no_clip:
        #     print("CLIP model is not initialized.")
        #     return
        # TODO: 展示时接入语义模块
        text_query = input("Enter your clip query: ")
        # text_queries = [text_query]
        
        text_feature = clip_model.encode_text(clip.tokenize(text_query).to("cuda"))
        text_feature /= text_feature.norm(dim=-1, keepdim=True)
        # similarities = objects.compute_similarities(text_query_ft)
        objects_clip_fts = objects.get_stacked_values_torch("clip_ft")
        objects_clip_fts = objects_clip_fts.to("cuda")
        similarities = F.cosine_similarity(
            text_feature.unsqueeze(0), objects_clip_fts, dim=-1
        )
        max_value = similarities.max()
        min_value = similarities.min()
        normalized_similarities = (similarities - min_value) / (max_value - min_value)
        probs = F.softmax(similarities, dim=0)      # 将每一列的相似性值映射到[0,1]区间内，得到概率分布
        max_prob_idx = np.argmax(similarities.detach().cpu().numpy())      # 转到numpy格式，返回最大值的元素索引
        similarity_colors = cmap(normalized_similarities.detach().cpu().numpy())[..., :3]   # 将相似性值映射为颜色
        similarity_colors = similarity_colors[0]
        print("max_prob_idx: ", max_prob_idx)
        # 更新点云对象的颜色属性，以反映每个对象的相似性
        for i in range(len(objects)):
            pcd = pcds[i]
            if i == max_prob_idx:
                points = np.asarray(pcd.points)
                print(i,points.mean(axis=0))
            map_colors = np.asarray(pcd.colors) # 先保存一下原有的颜色信息，但好像也没啥用
            # print(similarity_colors ,similarity_colors[i], similarity_colors[i, 0])
            pcd.colors = o3d.utility.Vector3dVector(        # 将numpy数组转换为open3d中的颜色向量格式
                np.tile(
                    [
                        similarity_colors[i, 0].item(),     # 第i行第一个元素，即红色通道的值，并得到标量值
                        similarity_colors[i, 1].item(),
                        similarity_colors[i, 2].item()
                    ], 
                    (len(pcd.points), 1)    # 使用np.tile函数将similarity_colors[i] 沿着第一个轴（行轴）复制成指定的行数，保证为每个点都分配相同的颜色
                )
            )

        for pcd in pcds:
            vis.update_geometry(pcd)
            
    def save_view_params(vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters("temp.json", param)
        
    vis.register_key_callback(ord("B"), toggle_bg_pcd)
    vis.register_key_callback(ord("C"), color_by_class)
    vis.register_key_callback(ord("R"), color_by_rgb)
    vis.register_key_callback(ord("F"), color_by_clip_sim)
    vis.register_key_callback(ord("T"), color_by_sbert_sim)
    vis.register_key_callback(ord("I"), color_by_instance)
    vis.register_key_callback(ord("V"), save_view_params)
    vis.register_key_callback(ord("N"), color_by_name)
    vis.register_key_callback(ord("Z"), color_by_chengzai_candidate)
    vis.register_key_callback(ord("K"), color_by_chengzai_small_obj)
    vis.register_key_callback(ord("W"), color_by_room_class)
    
    
    #! Render the scene
    vis.run()
    
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
