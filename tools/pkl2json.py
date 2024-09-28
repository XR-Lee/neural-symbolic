import json
import pickle
import argparse
import os
import numpy as np


def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(item) for item in obj]
    else:
        return obj

def pkl2json(pkl_data, output_json_path):
    data = pkl_data
    lanesegment_confidence_threshold = 0.5
    traffic_element_confidence_threshold = 0.5
    area_confidence_threshold = 0.5
    filtered_data = {'predictions': {'lane_segment': [], 'traffic_element': [],'area': [], 'topology_lsls': [], 'topology_lste': []},'sum_hw':{'h':{}, 'w':{}, 'a':{}}}
    
    centerlines =[]
    leftlines = []
    rightlines = []
    lefttypes = []
    righttypes = []
    confidences = []
    ids = []
    ls_index = []
    
    num1 = len(data['predictions']['lane_segment'])
    num2 = len(data['predictions']['traffic_element'])
    num3 = len(data['predictions']['area'])
    

    for lanecentrt in range(num1):
        centline = data['predictions']['lane_segment'][lanecentrt]['centerline']
        condfince = data['predictions']['lane_segment'][lanecentrt]['confidence']
        leftline = data['predictions']['lane_segment'][lanecentrt]['left_laneline']
        rightline = data['predictions']['lane_segment'][lanecentrt]['right_laneline']
        lefttype = data['predictions']['lane_segment'][lanecentrt]['left_laneline_type']
        righttype = data['predictions']['lane_segment'][lanecentrt]['right_laneline_type']
        id_pkl = data['predictions']['lane_segment'][lanecentrt]['id']
        
        if condfince > lanesegment_confidence_threshold:
            centerlines.append(centline)
            leftlines.append(leftline)
            rightlines.append(rightline)
            lefttypes.append(lefttype)
            righttypes.append(righttype)
            confidences.append(condfince)
            ids.append(id_pkl)
            ls_index.append(lanecentrt)

    traffic_elements = []
    traffic_elements_types = []
    te_index = []

    for traffic in range(num2):
        traffic_element = data['predictions']['traffic_element'][traffic]['points']
        traffic_element_type = data['predictions']['traffic_element'][traffic]['attribute']
        condfince = data['predictions']['traffic_element'][traffic]['confidence']

        if condfince > traffic_element_confidence_threshold:
            traffic_elements.append(traffic_element)
            traffic_elements_types.append(traffic_element_type)
            te_index.append(traffic)
    
    areas = []
    area_cates = []
    
    for area1 in range(num3):
        area = data['predictions']['area'][area1]['points']
        area_cate = data['predictions']['area'][area1]['category']
        condfince = data['predictions']['area'][area1]['confidence']
        
        if condfince > area_confidence_threshold:
            areas.append(area)
            area_cates.append(area_cate)

    top_ll = data['predictions']['topology_lsls'][np.ix_(ls_index, ls_index)]
    top_lt = data['predictions']['topology_lste'][np.ix_(ls_index, te_index)]
            
    len_lane = len(centerlines)
    len_traffic = len(traffic_elements)
    len_area = len(areas)

    for i in range(len_lane):
        lane_segment_data = {
            'id': convert_numpy(ids[i]),
            'centerline': convert_numpy(centerlines[i]),
            'left_laneline': convert_numpy(leftlines[i]),
            'right_laneline': convert_numpy(rightlines[i]),
            'left_laneline_type': convert_numpy(lefttypes[i]),
            'right_laneline_type': convert_numpy(righttypes[i]),
            'confidence': convert_numpy(confidences[i])
        }
        filtered_data['predictions']['lane_segment'].append(lane_segment_data)

    for j in range(len_traffic):
        traffic_element_data = {
            'points': convert_numpy(traffic_elements[j]),
            'attribute': convert_numpy(traffic_elements_types[j])
        }
        filtered_data['predictions']['traffic_element'].append(traffic_element_data)
        
    for k in range(len_area):
        area_data = {
            'points': convert_numpy(areas[k]),
            'category': convert_numpy(area_cates[k])
        }
        filtered_data['predictions']['area'].append(area_data)
        
    filtered_data['predictions']['topology_lsls'] = convert_numpy(top_ll)
    filtered_data['predictions']['topology_lste'] = convert_numpy(top_lt)
        
    numh = len(filtered_data['predictions']['lane_segment'])
    numw = len(filtered_data['predictions']['traffic_element'])
    numa = len(filtered_data['predictions']['area'])
    
    filtered_data['sum_hw']['h'] = num1    
    filtered_data['sum_hw']['w'] = num2
    filtered_data['sum_hw']['a'] = numa
    
    try:
        with open(output_json_path, 'w') as file:
            json.dump(filtered_data, file, indent=4)
    except Exception as e:
        print("Failed to write JSON file:", str(e))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='input path', default='./dataset/results_base.pkl')
    parser.add_argument('--output', help='output path', default='./dataset/output_json')
    parser.add_argument('--verbose', action='store_true', default=True)
    args = parser.parse_args()

    with open(args.input, 'rb') as file:
        pkl_data = pickle.load(file)

    result_dict = pkl_data['results']
    sum = 0
    for key in result_dict:
        scene = key[1]
        timestamp = key[2]
        save_path = f'{args.output}/{scene}/info/'
        if not os.path.exists(save_path): 
            os.makedirs(save_path)
        data_results = result_dict[key]
        output_json_path = f'{args.output}/{scene}/info/{timestamp}-ls.json'
        pkl2json(data_results, output_json_path)
        sum += 1
        if args.verbose:
            print(f'{sum} conversions finished.')
