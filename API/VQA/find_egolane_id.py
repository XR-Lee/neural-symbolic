import json
import os
from shapely.geometry import Point, Polygon

def check_point(x, y, x_less_than_zero, y_greater_than_zero):
    return (x < 0) if x_less_than_zero else (x > 0) and (y > 0 if y_greater_than_zero else y < 0)

def filter_ids(lane_segments):
    filtered_ids = []
    for segment in lane_segments:
        left_laneline = segment["left_laneline"]
        right_laneline = segment["right_laneline"]
        
        # chenck left_laneline first and last point
        if (check_point(left_laneline[0][0], left_laneline[0][1], True, True) and
            check_point(left_laneline[-1][0], left_laneline[-1][1], False, True)):
            
            # chenck right_laneline first and last point
            if (check_point(right_laneline[0][0], right_laneline[0][1], True, False) and
                check_point(right_laneline[-1][0], right_laneline[-1][1], False, False)):
                filtered_ids.append(segment["id"])
    return filtered_ids
    
def get_point(laneline):
    return (-laneline[1],laneline[0])
# def get_point(laneline):
#     return (laneline[0],laneline[1])

def ploygon_filter_ids(lane_segments):
    filtered_ids = filter_ids(lane_segments)
    if len(filtered_ids)!=0:
        return filtered_ids
    
    for i,segment in enumerate(lane_segments):
        left_laneline = segment["left_laneline"]
        right_laneline = segment["right_laneline"]


        # if (check_point(left_laneline[0][0], left_laneline[0][1], True, True) and
        #     check_point(left_laneline[-1][0], left_laneline[-1][1], False, True)):
            
        #     # chenck right_laneline first and last point
        #     if (check_point(right_laneline[0][0], right_laneline[0][1], True, False) and
        #         check_point(right_laneline[-1][0], right_laneline[-1][1], False, False)):
        #         filtered_ids.append(segment["id"])
        #         return filtered_ids


        if left_laneline[0][0] > left_laneline[-1][0]:
            continue
        
        # print('left_laneline[0]',left_laneline[0])
        # print('left_laneline[-1]',left_laneline[-1])
        # print('right_laneline[0]',right_laneline[0])
        # print('right_laneline[-1]',right_laneline[-1])

        

        left_bot=left_laneline[0]
        left_top=left_laneline[-1]
        right_bot=right_laneline[0]
        right_top=right_laneline[-1]

        # right_bot=left_laneline[0]
        # left_bot=left_laneline[-1]
        # right_top=right_laneline[0]
        # left_top=right_laneline[-1]

        left_bot=get_point(left_bot)
        left_top=get_point(left_top)
        right_bot=get_point(right_bot)
        right_top=get_point(right_top)

        polygon = Polygon([left_bot, left_top, right_bot, right_top])
        point = Point(0, 0)
        distance = point.distance(polygon)

        if 'min_distance' not in locals():
            min_distance=distance
            filtered_ids.append(segment["id"])

        if distance < min_distance:
            min_distance=distance
            filtered_ids.pop()
            filtered_ids.append(segment["id"])
            

    if len(filtered_ids) ==0:
        print("errrrrr")
    return filtered_ids

if __name__ == '__main__':

    directory_path = '/DATA_EDS2/zhangzz2401/zhangzz2401/OpenLane-V2-master/mapless/pkl2json_mini_batch'

    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)

            parts = file_path.split('/')
            scene_id = parts[-3]
            sample_id = parts[-1]
            timestamp = sample_id.split('-')[0]

            print(scene_id)
            print(timestamp)

            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            search_lane_id = ploygon_filter_ids(data["predictions"]["lane_segment"])
            print("ego_lane_id :", search_lane_id)
