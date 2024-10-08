from write_location_lanes import parse_file , find_file
import json

filename = '/DATA_EDS2/zhangzz2401/zhangzz2401/OpenLane-V2-master/mapless/ego_para_area_result_complete.txt'  
directory = '/DATA_EDS2/zhangzz2401/zhangzz2401/OpenLane-V2-master/output_topomlp_complete'
with open(filename, 'r') as file:
    lines =file.readlines()
    for index in range(len(lines)):
        data = parse_file(filename, index)
        scene_id = data[0]
        timestamp = data[1]
        lane_id = data[2]
        intersection = data[3]

        if intersection == 0:
            continue
        else:
            found_file = find_file(directory, scene_id, timestamp)
            print(found_file)
            with open(found_file, 'r') as file:
                json_data = json.load(file)
                for information in json_data["information"]:
                    if information['lane_id']==lane_id:
                        information['intersection']='1'

                        with open(found_file, 'w') as file:  
                            json.dump(json_data, file, indent=4) 
                        
                        break

