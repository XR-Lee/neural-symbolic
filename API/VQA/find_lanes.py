import json
import os
from find_egolane_id import ploygon_filter_ids
from find_parallel_lanes import search_para_segment




if __name__ == '__main__':
    directory_path = '/DATA_EDS2/zhangzz2401/zhangzz2401/OpenLane-V2-master/mapless/pkl2json_results_base'

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
            para_lane_ids=[]
            for id in search_lane_id:
                para_lane_ids.extend(search_para_segment(scene_id, timestamp, str(id)))
            print("search_lane_id :", search_lane_id)
            print("para_lane_ids :", para_lane_ids)
            ids = list(set(search_lane_id+para_lane_ids))
            print("ids :", ids)
