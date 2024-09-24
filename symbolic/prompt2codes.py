import os
import pickle
import copy
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from tqdm import tqdm
# from eval import evaluate

from pydantic import BaseModel
import openai  
# from typing import List
import re
from openai import OpenAI

client = OpenAI(
  api_key="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",  # this is also the default, it can be omitted
)


# Make the API call to get the completion
completion = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",  # Adjust to the correct model name
    messages=[
        {"role": "system", "content": "Generate the python function for direct excutable, no example."},
        {"role": "user", "content": "You are an expert in processing lanes in autonomous driving,\
            here you need to write a condition function to judge if two lanes are connected or not.\
            the fucntion named: condition should take only (ref_centerline, cmp_centerline,ref_leftline,\
            ref_rightline, cmp_leftline, cmp_rightline) as input and outputs True or False. all lines\
            are list of points (x, y)  with order.  based on the following domain expert rules: \
            1.the distance between the ref line last point  and the start point of cmp line should \
            not be too far, you need to reason the distance threshold by yourself step by step,  \
            if too far return True, else False "},
    ],
)

# Parse the response content (you'll need to handle this according to the actual API response structure)

print( completion.choices[0].message.content)


gpt_response = completion.choices[0].message.content
# Use regex to extract code between the ```python ... ```
code_block = re.search(r'```python(.*?)```', gpt_response, re.DOTALL)

# If a code block is found, extract and clean the code
if code_block:
    python_code = code_block.group(1).strip()
    print("Extracted Python Code:")
    print(python_code)
else:
    print("No Python code block found.")
    
    


format_res_path = '/fs/scratch/Sgh_CR_RIX/rix3_shared/win_shared/0530_results_ep34_formatted.pkl'
format_gt_path = "/fs/scratch/Sgh_CR_RIX/rix3_shared/win_shared/val_gt.pkl.pkl"

def transform_coord(line):
    return np.array([[-point[1], point[0]] for point in line.tolist()], dtype=np.float32)

# pred_dict = pickle.load(open(format_res_path, 'rb'))

# data = copy.deepcopy(pred_dict)
# pair_cnt = 0
# max_pair = 0
# first_key = list(data['results'].keys())[0]
# for key in tqdm([first_key], desc='Iteration'):
#     predictions = data['results'][key]['predictions']
#     lane_segments = predictions['lane_segment']
#     topll_matrix = predictions['topology_lsls']
#     mask_ids = []
#     for ref_lane in lane_segments:
#         mask_id = []
#         ref_centerline = transform_coord(ref_lane['centerline'])
#         ref_leftline = transform_coord(ref_lane['left_laneline'])
#         ref_rightline = transform_coord(ref_lane['right_laneline'])
#         for index, cmp_lane in enumerate(lane_segments):
#             cmp_centerline = transform_coord(cmp_lane['centerline'])
#             cmp_leftline = transform_coord(cmp_lane['left_laneline'])
#             cmp_rightline = transform_coord(cmp_lane['right_laneline'])
#             #   and lr_trigger(ref_leftline, ref_rightline, cmp_leftline, cmp_rightline)
#             # if condition(ref_centerline, cmp_centerline,ref_leftline, ref_rightline, cmp_leftline, cmp_rightline):
#             mask_id.append(index)
#             pair_cnt += 1
#             # elif lr_trigger(ref_leftline, ref_rightline, cmp_leftline, cmp_rightline):
#             #     mask_id.append(index)
#             #     pair_cnt += 1

#         max_pair = max(max_pair, len(mask_id))
#         mask_ids.append(mask_id)
    
#     for i in range(len(mask_ids)):
#         for j in range(len(mask_ids)):
#             if j in mask_ids[i]:
#                 topll_matrix[i][j] = 1
#             else:
#                 topll_matrix[i][j] = 0
#     data['results'][key]['predictions']['topology_lsls'] = topll_matrix
#     print(topll_matrix)

base_code = '''
pred_dict = pickle.load(open(format_res_path, 'rb'))
data = copy.deepcopy(pred_dict)
pair_cnt = 0
max_pair = 0
first_key = list(data['results'].keys())[0]
for key in tqdm([first_key], desc='Iteration'):
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
            if condition(ref_centerline, cmp_centerline,ref_leftline, ref_rightline, cmp_leftline, cmp_rightline):
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
    print(topll_matrix)
'''


final_codes = python_code +"\n"+ base_code


print(final_codes)
exec(final_codes)

# mean_pair_cnt = pair_cnt / len(data['results'].keys())
# print("Angle threshold: ", angle_thres)
# print("Max pair count: ", max_pair)
# print("Mean pair count: ", mean_pair_cnt)
# res =  evaluate(gt_dict, data, verbose=True)
# print(res)

# pickle.dump(data, open(save_path, "wb"))
