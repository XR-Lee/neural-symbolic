import numpy as np
import json
import os

import json
def create_json(A, B, pth):  
    data = {  

        "segment_id": str(A),  
        "timestamp": str(B) ,
        "information":[],
    }  

    pth=pth+A+'_'+B+'.json'
      
    with open(pth, 'w') as json_file:  

        json.dump(data, json_file, indent=4)  

def write_information(search_lane_id, connected_lane_id, pth):  

    with open(pth, 'r') as file:  
        data = json.load(file)  

    if len(data["information"])!=0 and data["information"][-1]["lane_id"]==str(search_lane_id):
        if connected_lane_id!='' :
            data["information"][-1]["connection"] +=(''if data["information"][-1]["connection"]==''else',')+str(connected_lane_id)
    else:
        new_info = {  
            "lane_id": str(search_lane_id),  
            "intersection": "0",
            "connection": str(connected_lane_id),
            "location": ""
        }  

        data["information"].append(new_info)  

    with open(pth, 'w') as file:  

        json.dump(data, file, indent=4)  

directory_path = '/DATA_EDS2/zhangzz2401/zhangzz2401/OpenLane-V2-master/mapless/pkl2json_results_base'
#output_json_filename = '/DATA_EDS2/zhangzz2401/zhangzz2401/OpenLane-V2-master/mapless/connection.json'

for root, dirs, files in os.walk(directory_path):
    for file in files:
        if file.endswith('.json'):
            file_path = os.path.join(root, file)
            print(file_path)
        parts = file_path.split('/')

        #segment_id, timestamp
        segment_id = parts[-3]
        sample_id = parts[-1]
        timestamp = sample_id.split('-')[0]
        create_json(segment_id,timestamp,"./output_topomlp_complete/")
        # print(segment_id, timestamp)  

        with open(file_path, 'r') as file:
            data = json.load(file)

        lsls_matrix = data["predictions"]["topology_lsls"]
        lane_segments = data["predictions"]["lane_segment"]

        binary_lsls_matrix = [[1 if val >= 0.5 else 0 for val in arr] for arr in lsls_matrix]

        num_rows = len(binary_lsls_matrix)
        if num_rows == 0:
            num_columns = num_rows
        else:
            num_columns = len(binary_lsls_matrix[0])

        print(num_rows, num_columns)
        
        i = 0
        j = 0

        for i in range(num_rows):
            search_lane_id = lane_segments[i].get("id")
            for j in range(num_columns):
                aij = binary_lsls_matrix[i][j]
                connected_lane_id = ""
                if aij == 1:
                    connected_lane_id = lane_segments[j].get("id")
                    # print(search_lane_id, connected_lane_id)
                
                    # print('------------------------------------------------------------')
                write_information(search_lane_id,connected_lane_id,'./output_topomlp_complete/'+segment_id+'_'+timestamp+'.json')


  
