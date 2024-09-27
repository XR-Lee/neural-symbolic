import os
import json
import cv2
import numpy as np


# jsons_path = '/DATA_EDS2/zhangzz2401/zhangzz2401/OpenLane-V2-master/mapless/pkl2json_mini_batch/'
jsons_path = '/DATA_EDS2/zhangzz2401/zhangzz2401/OpenLane-V2-master/mapless/pkl2json_results'

def transform_coord(line):
    return np.array([[-point[1], point[0]] for point in line], dtype=np.float32)

def dilate_vector(ori, vector, length, num_segments=20):
    dis = np.linalg.norm(vector)
    id_vector = vector / dis
    step = length / num_segments
    sub_vector = id_vector * step

    return np.array([ori + i * sub_vector for i in range(3, num_segments + 3)], dtype=np.float32)

def is_polygons_overlap(polygon1, polygon2):
    is_overlap, _ = cv2.intersectConvexConvex(polygon1, polygon2)
    
    return is_overlap

def is_opposite_lane(centerline):
    return centerline[-1][1] - centerline[0][1] < 0

def search_para_segment(scene, timestamp, ego_lane_id):
    json_path = os.path.join(jsons_path, scene, 'info', timestamp) + '-ls.json'
    with open(json_path) as f:
        json_data = json.load(f)

    lane_ids = []
    centerlines = []
    leftlines = []
    rightlines = []
    for i in range(len(json_data["predictions"]['lane_segment'])):
        lane_id = json_data["predictions"]['lane_segment'][i]['id']
        centerline = json_data["predictions"]['lane_segment'][i]['centerline']
        leftline = json_data["predictions"]['lane_segment'][i]['left_laneline']
        rightline = json_data["predictions"]['lane_segment'][i]['right_laneline']
        lane_ids.append(lane_id)
        centerlines.append(centerline)
        leftlines.append(leftline)
        rightlines.append(rightline)
    
    result_ids = []
    ego_idx = lane_ids.index(int(ego_lane_id))
    ego_leftline = transform_coord(leftlines[ego_idx])
    ego_rightline = transform_coord(rightlines[ego_idx])
    dilated_left_seg = dilate_vector(ego_leftline[len(ego_leftline)//2], ego_leftline[len(ego_leftline)//2] - ego_rightline[len(ego_rightline)//2], 10)
    dilated_right_seg = dilate_vector(ego_rightline[len(ego_rightline)//2], ego_rightline[len(ego_rightline)//2] - ego_leftline[len(ego_leftline)//2], 10)
    
    for index, (centerline, leftline, rightline) in enumerate(zip(centerlines, leftlines, rightlines)):
        if lane_ids[index] != ego_lane_id:
            cmp_leftline = transform_coord(leftline)
            cmp_rightline = transform_coord(rightline)
            cmp_centerline = transform_coord(centerline)
            cmp_polygon = np.concatenate((cmp_leftline, cmp_rightline[::-1]), axis=0)
            if is_polygons_overlap(dilated_left_seg, cmp_polygon) or is_polygons_overlap(dilated_right_seg, cmp_polygon):
                if not is_opposite_lane(cmp_centerline):
                    result_ids.append(lane_ids[index])
        
    return result_ids

if __name__ == '__main__':
    scene = "10002"
    timestamp = '315971486549927216'
    ego_lane_id = '20013'
    para_lane_ids = search_para_segment(scene, timestamp, ego_lane_id)
    print(scene, timestamp, para_lane_ids)