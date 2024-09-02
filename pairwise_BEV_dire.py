import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
import copy
from shapely.geometry import Polygon, Point
import os
import json
from PIL import Image, ImageDraw
import logging
import math
import time

num_img = 0

def transform_coord(line):
    return np.array([[-point[1], point[0]] for point in line], dtype=np.float32)

def dilate_seg(centerline, length, num_segments=20):
    vector = centerline[-1] - centerline[-2]
    dis = np.linalg.norm(vector)
    id_vector = vector / dis
    step = length / num_segments
    sub_vector = id_vector * step

    return np.array([centerline[-1] + i * sub_vector for i in range(3, num_segments + 3)], dtype=np.float32)

def reorder_polygon(polygon, center_point=None):
    if center_point is None:
        center_point = polygon.mean(axis=0)
    
    ref_point = polygon[0]
    ref_vector = ref_point - center_point
    angle_list = [(get_vector_angle(ref_vector, point - center_point), idx) for idx, point in enumerate(polygon)]
    sorted_angle_list = sorted(angle_list, key=lambda x: x[0])
    index_list = [pair[1] for pair in sorted_angle_list]

    return polygon[index_list]

def is_polygons_overlap(polygon1, polygon2):
    is_overlap, intersection = cv2.intersectConvexConvex(polygon1, polygon2)
    # if intersection is not None:
    #     intersection = reorder_polygon(intersection.squeeze(axis=1))
    
    return is_overlap, intersection


def polygon_area(polygon):
    polygon = Polygon(polygon.tolist())
    area = polygon.area
    return area

def calculate_iou(polygon1, polygon2):
    is_overlap, intersection = is_polygons_overlap(polygon1, polygon2)
    
    if not is_overlap:
        return 0
    else:
        intersection = intersection.squeeze(axis=1)
    
    area_intersection = polygon_area(intersection)
    area_polygon1 = polygon_area(polygon1)
    area_polygon2 = polygon_area(polygon2)

    # print(area_intersection)
    # print(area_polygon1)
    # print(area_polygon2)
    # time.sleep(2)
    
    area_union = area_polygon1 + area_polygon2 - area_intersection
    
    iou = area_intersection / area_union

    return iou

def get_vector_angle(ref_vector, cmp_vector):
    dot_product = np.dot(ref_vector, cmp_vector)
    ref_norm = np.linalg.norm(ref_vector)
    cmp_norm = np.linalg.norm(cmp_vector)
    cos_theta = dot_product / (ref_norm * cmp_norm)
    cos_theta = np.clip(cos_theta, -1, 1)
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad) + 360
    while angle_deg > 360:
        angle_deg -= 360
    
    return angle_deg

def calculate_angle(ref_centerline, cmp_centerline):
    ref_vector = ref_centerline[-1] - ref_centerline[-2]
    cmp_vector = cmp_centerline[1] - cmp_centerline[0]
    dot_product = np.dot(ref_vector, cmp_vector)
    ref_norm = np.linalg.norm(ref_vector)
    cmp_norm = np.linalg.norm(cmp_vector)
    cos_theta = dot_product / (ref_norm * cmp_norm)
    cos_theta = np.clip(cos_theta, -1, 1)
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    if angle_deg > 180:
        angle_deg = 360 - angle_deg
    
    return angle_deg

def get_iou(ref_id, ref_polygon, ver_polygons):
   return [calculate_iou(ref_polygon, polygon) for index, polygon in enumerate(ver_polygons) if index != ref_id]

def draw_pair_BEV(leftlines, rightlines, ref_polygon, ver_polygons, ver_idx, name):
    for index, polygon in enumerate(ver_polygons):
        fig, ax = plt.subplots(figsize=(10, 20))
        ax.set_axis_off()  # ???????
        fig.patch.set_facecolor('black')
        for lanes in leftlines:  ##11??
            #lanes = lanes[int(len(centerline) * 0.02):int(len(centerline) * 0.98)]
            
            centerline_bevx = []
            centerline_bevy = []            
            
            for point in lanes:
                x, y, z = point
                new_x = 1 * x
                new_y = -1 * y
                centerline_bevx.append(new_y)   ##?????????  90?
                centerline_bevy.append(new_x)

            ax.plot(centerline_bevx, centerline_bevy, color='red', linewidth=1.5, zorder=5)
    

        for lanes in rightlines:  ##11??
            #lanes = lanes[int(len(centerline) * 0.02):int(len(centerline) * 0.98)]
            
            centerline_bevx = []
            centerline_bevy = []            
            
            for point in lanes:
                x, y, z = point
                new_x = 1 * x
                new_y = -1 * y
                centerline_bevx.append(new_y)   ##?????????  90?
                centerline_bevy.append(new_x)

            ax.plot(centerline_bevx, centerline_bevy, color='red', linewidth=1.5, zorder=5)
        
        green_polygon = ref_polygon.tolist()
        green_polygon = patches.Polygon(green_polygon, closed=True, fill=True, color='green', zorder=10)
        ax.add_patch(green_polygon)
        blue_polygon = polygon.tolist()
        # polygon = [(point[0], point[1]) for point in polygon]
        # print(name + f'-inter-{idx}.png')
        # print(intersection.tolist())
        # time.sleep(3)
        blue_polygon = patches.Polygon(blue_polygon, closed=True, fill=True, color='blue', zorder=5)
        ax.add_patch(blue_polygon)

        # aspect_ratio = fig.get_figwidth() / fig.get_figheight()
        # new_width = 512
        # new_height = int(new_width / aspect_ratio)
        plt.savefig(name + f'-{ver_idx[index]}.png', dpi=66, bbox_inches='tight', pad_inches=0, facecolor='black')
        plt.close(fig)
        global num_img
        num_img += 1

def lr_trigger(ref_leftline, ref_rightline, cmp_leftline, cmp_rightline):
    ref_left_point = Point(ref_leftline[-1])
    ref_right_point = Point(ref_rightline[-1])
    cmp_left_point = Point(cmp_leftline[0])
    cmp_right_point = Point(cmp_rightline[0])
    left_distance = ref_left_point.distance(cmp_left_point)
    right_distance = ref_right_point.distance(cmp_right_point)

    return left_distance < 1.5 and right_distance < 1.5

def IMG(jsonname,jsonname_new):####.....json
    print('----------')
    print(jsonname)
    # ?? JSON ??    ##'/DATA_EDS2/zhangzz2401/zhangzz2401/OpenLane-V2-master/data/OpenLane-V2/val/10000/info/xxx-ls.json'
    root,pathname,imgname = jsonname.split('/')[-3],jsonname.split('/')[-1],jsonname.split('val')[0]  ##???10000?json???  /DATA_EDS2/zhangzz2401/zhangzz2401/OpenLane-V2-master/data/OpenLane-V2/
    path = pathname.split('.')[0]   

    with open(jsonname) as f:
        data = json.load(f)
        
    with open(jsonname_new) as f:
        data_new = json.load(f)


    # ??????? 
    centerlines = []      ### [0,10]
    leftlines = []
    rightlines = []
    lefttypes = []
    righttypes = []
    for i in range(len(data_new["predictions"]['lane_segment'])):
        centerline = data_new["predictions"]['lane_segment'][i]['centerline']
        leftline = data_new["predictions"]['lane_segment'][i]['left_laneline']
        rightline = data_new["predictions"]['lane_segment'][i]['right_laneline']
        lefttype = data_new["predictions"]['lane_segment'][i]['left_laneline_type']
        righttype = data_new["predictions"]['lane_segment'][i]['right_laneline_type']
        centerlines.append(centerline)
        leftlines.append(leftline)
        rightlines.append(rightline)
        lefttypes.append(lefttype)
        righttypes.append(righttype)
    


    intrinsic = data["sensor"]['ring_front_center']['intrinsic']
    extrinsic = data["sensor"]['ring_front_center']['extrinsic']   ##???? ?? 
    

    if "traffic_element" in data_new["predictions"]:
        LT = data_new["predictions"]['traffic_element']    ###?????

    num = len(centerlines)  # ????
    num_lt = len(LT) if "traffic_element" in data_new["predictions"] else 0  # ??????
    
    image = data['sensor']['ring_front_center']['image_path']   
    imagename = imgname + image ##????

    
    k_points = []
    attribute = []
    for i in range(len(data_new["predictions"]['traffic_element'])):
        point = data_new["predictions"]['traffic_element'][i]['points']
        attr = data_new["predictions"]['traffic_element'][i]['attribute']
        k_points.append(point)
        attribute.append(attr)
                
    k_points = np.array(k_points)

    
    
    area_points = []
    area_cates = []  
    for i in range(len(data_new["predictions"]['area'])):
        point = data_new["predictions"]['area'][i]['points']
        area_cate = data_new["predictions"]['area'][i]['category']
        area_points.append(point)
        area_cates.append(area_cate)
        
    num_area = len(data_new["predictions"]['area'])
    colors = [(1, 0, 0), (0, 0, 1), (0, 1, 0)]
   
    file = '/DATA_EDS2/wanggl/datasets/BEV_pairwise_polygon_dire_lr_true_1.5'
    rootimg = file + '/' + root + '/' 
    if not os.path.exists(rootimg):
        os.makedirs(rootimg)
    
    # IoU_thres = 0.12
    angle_thres = 90
    for index1, (cent1, left1, right1) in enumerate(zip(centerlines, leftlines, rightlines)):
        # pair_cnt = 0
        # ver_centerlines = []
        # ver_polygons = []
        # ver_idx = []
        ref_leftline = transform_coord(left1)
        ref_rightline = transform_coord(right1)
        ref_centerline = transform_coord(cent1)
        # ref_polygon = np.concatenate((ref_leftline, ref_rightline[::-1]), axis=0)
        # dilated_seg = dilate_seg(ref_centerline, 5)
        name = f'{file}/{root}/{path}-{index1}'
        for index2, (cent2, left2, right2) in enumerate(zip(centerlines, leftlines, rightlines)):
            if index1 != index2:
                cmp_centerline = transform_coord(cent2)
                cmp_leftline = transform_coord(left2)
                cmp_rightline = transform_coord(right2)
                # cmp_polygon = np.concatenate((cmp_leftline, cmp_rightline[::-1]), axis=0)
                # if is_polygons_overlap(dilated_seg, cmp_polygon)[0]:
                if calculate_angle(ref_centerline, cmp_centerline) < angle_thres and lr_trigger(ref_leftline, ref_rightline, cmp_leftline, cmp_rightline):
                    # ver_centerlines.append(cmp_centerline)
                    # ver_polygons.append(cmp_polygon)
                    # ver_idx.append(index2)
                    # if pair_cnt == 0:
                    # left_distance, right_distance = lr_trigger(ref_leftline, ref_rightline, cmp_leftline, cmp_rightline)[1:]
                    ##BEV
                    fig, ax = plt.subplots(figsize=(10, 20))
                    ax.set_axis_off()  # ???????
                    fig.patch.set_facecolor('black')
                
                    for lanes in leftlines:  ##11??
                        #lanes = lanes[int(len(centerline) * 0.02):int(len(centerline) * 0.98)]
                        
                        centerline_bevx = []
                        centerline_bevy = []            
                        
                        for point in lanes:
                            x, y, z = point
                            new_x = 1 * x
                            new_y = -1 * y
                            centerline_bevx.append(new_y)   ##?????????  90?
                            centerline_bevy.append(new_x)

                        ax.plot(centerline_bevx, centerline_bevy, color=colors[0], linewidth=1.5, zorder=5)
                

                    for lanes in rightlines:  ##11??
                        #lanes = lanes[int(len(centerline) * 0.02):int(len(centerline) * 0.98)]
                        
                        centerline_bevx = []
                        centerline_bevy = []            
                        
                        for point in lanes:
                            x, y, z = point
                            new_x = 1 * x
                            new_y = -1 * y
                            centerline_bevx.append(new_y)   ##?????????  90?
                            centerline_bevy.append(new_x)

                        ax.plot(centerline_bevx, centerline_bevy, color=colors[0], linewidth=1.5, zorder=5)

                    points1 = []
                    for point in left1:
                        x, y, z = point
                        new_x = 1 * x
                        new_y = -1 * y
                        points1.append((new_y, new_x))
                    
                    for point in reversed(right1):
                        x, y, z = point
                        new_x = 1 * x
                        new_y = -1 * y
                        points1.append((new_y, new_x))

                    lane_x1 = []
                    lane_y1 = []
                    for point in cent1:
                        x, y, z = point
                        new_x = 1 * x
                        new_y = -1 * y
                        lane_x1.append(new_y)
                        lane_y1.append(new_x)

                    cent_point1 = []
                    x, y, z = cent1[-2]
                    new_x = 1 * x
                    new_y = -1 * y
                    cent_point1.append((new_y, new_x))
                    x, y, z = cent1[-1]
                    new_x = 1 * x
                    new_y = -1 * y
                    cent_point1.append((new_y, new_x))

                    polygon = patches.Polygon(points1, closed=True, fill=True, color='green', zorder=10)
                    ax.add_patch(polygon)

                        # ax.plot([dilated_seg[0][0], dilated_seg[-1][0]], [dilated_seg[0][1], dilated_seg[-1][1]], color='white', linewidth=3, zorder=5)
            
                    # pair_cnt += 1
                    # name += f'-{index2}'
                    # palette = ['blue', 'yellow', 'white', 'magenta', 'cyan', 'orange', 'crimson', 'brown', 'silver', 'golden']
                    points2 = []
                    for point in left2:
                        x, y, z = point
                        new_x = 1 * x
                        new_y = -1 * y
                        points2.append((new_y, new_x))
                    
                    for point in reversed(right2):
                        x, y, z = point
                        new_x = 1 * x
                        new_y = -1 * y
                        points2.append((new_y, new_x))

                    lane_x2 = []
                    lane_y2 = []
                    for point in cent2:
                        x, y, z = point
                        new_x = 1 * x
                        new_y = -1 * y
                        lane_x2.append(new_y)
                        lane_y2.append(new_x)

                    cent_point2 = []
                    x, y, z = cent2[-2]
                    new_x = 1 * x
                    new_y = -1 * y
                    cent_point2.append((new_y, new_x))
                    x, y, z = cent2[-1]
                    new_x = 1 * x
                    new_y = -1 * y
                    cent_point2.append((new_y, new_x))
                    
                    polygon = patches.Polygon(points2, closed=True, fill=True, color="blue")
                    ax.add_patch(polygon)
                    plt.savefig(name + f'-{index2}.png', bbox_inches='tight', pad_inches=0, facecolor='black')
                    plt.close(fig)
                    # print(name + f'-{index2}.png')
                    # print(left_distance, right_distance)
                    # time.sleep(2)
                    global num_img
                    num_img += 1

        # print(ver_polygons) any(angle > angle_thres for angle in angle_list)
        # print(len(ver_polygons)) any(iou < IoU_thres for iou in IoU_list)
        # IoU_filter_idx = [index for index, polygon in enumerate(ver_polygons) if any(iou < IoU_thres for iou in get_iou(index, polygon, ver_polygons))]
        # IoU_list = [calculate_iou(polygon1, polygon2) for index1, polygon1 in enumerate(ver_polygons) for index2, polygon2 in enumerate(ver_polygons) if index1 != index2]
        # angle_list = [calculate_angle(ref_centerline, ver_centerline) for ver_centerline in ver_centerlines]
        # if pair_cnt >= 2 and any(angle > angle_thres for angle in angle_list):
        #     draw_pair_BEV(leftlines, rightlines, ref_polygon, ver_polygons, ver_idx, name)
        # if pair_cnt >= 2 and any(iou < IoU_thres for iou in IoU_list):
        #     name += '.png'
        #     plt.savefig(name, bbox_inches='tight', pad_inches=0, facecolor='black')
        #     plt.close(fig)
        #     global num_img
        #     num_img += 1
        # elif pair_cnt > 0:
        #     plt.close(fig)
        # if root == '10004':
        #     print(IoU_list)
        #     time.sleep(3)
        # break
            
        

def main():
    
    rootpath = '/DATA_EDS2/zhangzz2401/zhangzz2401/OpenLane-V2-master/data/OpenLane-V2/val/'
    # rootpath_new = '/DATA_EDS2/wanggl/datasets/pkl2json72te/'
    rootpath_new = '/DATA_EDS2/zhangzz2401/zhangzz2401/OpenLane-V2-master/mapless/pkl2json_mini_batch/'

    for i in range(10000,10150):
        rootpath2 = rootpath + str(i).zfill(5) + '/' + 'info'   ##'/DATA_EDS2/zhangzz2401/zhangzz2401/OpenLane-V2-master/data/OpenLane-V2/train/10000/info'
        rootpath_new2 = rootpath_new + str(i).zfill(5) + '/' + 'info'
        for b in os.listdir(rootpath_new2):      
            rootpath3 = rootpath2 + '/' + b 
            rootpath_new3 = rootpath_new2 + '/' + b   
            namepart = rootpath3.split("-")
            if namepart[-1] == "ls.json":
                IMG(rootpath3,rootpath_new3)
            # break
        print(f'The {i} epoch done')
        # break
    print("Done!")
    print(num_img)
    

if __name__ == '__main__':
    main()
