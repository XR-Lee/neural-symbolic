import os
import pickle
import cv2
import copy
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from tqdm import tqdm
from eval import evaluate

format_res_path = '/DATA_EDS2/zhangzz2401/zhangzz2401/OpenLane-V2-master/mapless/submission.pkl'
format_gt_path = "/DATA_EDS2/zhangzz2401/zhangzz2401/OpenLane-V2-master/mapless/topll_gt.pkl"
# save_path = '/DATA_EDS2/zhangzz2401/zhangzz2401/OpenLane-V2-master/mapless/seg_based_submission.pkl'

def transform_coord(line):
    return np.array([[-point[1], point[0]] for point in line.tolist()], dtype=np.float32)

def lr_trigger(ref_leftline, ref_rightline, cmp_leftline, cmp_rightline):
    ref_left_point = Point(ref_leftline[-1])
    ref_right_point = Point(ref_rightline[-1])
    cmp_left_point = Point(cmp_leftline[0])
    cmp_right_point = Point(cmp_rightline[0])
    left_distance = ref_left_point.distance(cmp_left_point)
    right_distance = ref_right_point.distance(cmp_right_point)

    return left_distance < 2.5 and right_distance < 2.5

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


pred_dict = pickle.load(open(format_res_path, 'rb'))
gt_dict = pickle.load(open(format_gt_path, 'rb'))
angle_thres = 90

data = copy.deepcopy(pred_dict)
pair_cnt = 0
max_pair = 0
for key in tqdm(data['results'].keys(), desc='Iteration'):
    predictions = data['results'][key]['predictions']
    lane_segments = predictions['lane_segment']
    topll_matrix = predictions['topology_lsls']
    mask_ids = []
    for ref_lane in lane_segments:
        mask_id = []
        ref_centerline = transform_coord(ref_lane['centerline'])
        ref_leftline = transform_coord(ref_lane['left_laneline'])
        ref_rightline = transform_coord(ref_lane['right_laneline'])
        for index, cmp_lane in enumerate(lane_segments):
            cmp_centerline = transform_coord(cmp_lane['centerline'])
            cmp_leftline = transform_coord(cmp_lane['left_laneline'])
            cmp_rightline = transform_coord(cmp_lane['right_laneline'])
            #   and lr_trigger(ref_leftline, ref_rightline, cmp_leftline, cmp_rightline)
            if calculate_angle(ref_centerline, cmp_centerline) < angle_thres and lr_trigger(ref_leftline, ref_rightline, cmp_leftline, cmp_rightline):
                mask_id.append(index)
                pair_cnt += 1
            # elif lr_trigger(ref_leftline, ref_rightline, cmp_leftline, cmp_rightline):
            #     mask_id.append(index)
            #     pair_cnt += 1

        max_pair = max(max_pair, len(mask_id))
        mask_ids.append(mask_id)
    
    for i in range(len(mask_ids)):
        for j in range(len(mask_ids)):
            if j in mask_ids[i]:
                topll_matrix[i][j] = 1
            else:
                topll_matrix[i][j] = 0
    data['results'][key]['predictions']['topology_lsls'] = topll_matrix

mean_pair_cnt = pair_cnt / len(data['results'].keys())
print("Angle threshold: ", angle_thres)
print("Max pair count: ", max_pair)
print("Mean pair count: ", mean_pair_cnt)
res =  evaluate(gt_dict, data, verbose=True)
print(res)

# pickle.dump(data, open(save_path, "wb"))
