import openai
import os
import base64
import json
from IPython.display import Markdown, display
import time

os.environ['HTTP_PROXY'] = 'http://10.0.0.15:11452'
os.environ['HTTPS_PROXY'] = 'http://10.0.0.15:11452'

# ??OpenAI API??
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key= "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"
# ?????????????
# system_intel = '''
# You are an automated expert system specialized in analyzing road scenes for autonomous driving applications. Your task is to examine the relationship between two specified targets in a road scene: target B (a traffic element) and target A (lane line). You need to determine whether there is a topological relationship between them, defined by the decision-making influence of target B on target A.

# A topological relationship exists if target B influences the decision-making relevant to target A. For example, in a scenario where a road is divided into three parallel lanes (left, middle, right), and a sign prohibiting right turns is placed in front of these lanes, the sign would have a topological relationship only with the rightmost lane and not with the other two lanes.

# Task Description:
# - Assess the provided scene.
# - Identify the positions and characteristics of targets A and B.
# - Evaluate whether target B has a decision-making influence on target A based on their positions and the context of the road scene.
# - Output '1' if a topological relationship exists between target B and target A, or '0' if there is no such relationship.
# - Please use the provided data and rules to accurately judge the topological relationship in each scenario presented to you.

# For the coordinates and image content you receive, you need to conduct the following analysis:
# 1. **Lane Position Analysis**:
#    - Determine the relative position of the selected lane line in the image. Assess whether the lane line is positioned on the left, middle, or right side of the road. Output this positional information as part of your analysis.

# 2. **Traffic Element Category Analysis**:
#    - Evaluate the category of the traffic element to determine its specific influence:
#      - Elements such as 'left turn', 'no left turn', 'u-turn', and 'no u-turn' typically only influence the leftmost lane and should therefore have a topological relationship only with that lane.
#      - Similarly, elements such as 'right turn' and 'no right turn' generally pertain only to the rightmost lane.
#      - For other signs not mentioned above, you need to use your reasoning and judgment to make specific determinations. You need to reason about whether there is a topological relationship between them.

# 3. **Final Determination**:
#    - Based on the above analyses, determine if the identified topological relationship is valid:
#      - If the lane position and the type of traffic element correspond correctly (e.g., a 'left turn' sign in front of the leftmost lane), output '1' indicating a valid topological relationship.
#      - For other cases, the final result should be derived through the reasoning of the larger model based on the actual situation in the image.

# The final output format should strictly follow this structure:
# - step1: left or middle or right (lane position)
# - step2: left turn or no left turn, etc. (traffic element category)
# - step3: 0 or 1 (final topological relationship determination)

# Your decisions should be precise and based on a comprehensive analysis of both the visual data from the image and the logical rules applied. Ensure to process each sample diligently to maintain accuracy in the final output.

# '''

# system_intel = '''
# The red lines in the photos are lane boundaries. For example, we consider the lane segments in the more left lanes are on the right of those in the more right lanes. We determine their left-right positional relations through lanes instead of the absolute positions of segment patches.
# '''
system_intel = '''
In the provided bird's-eye view (BEV), the red lines in the photos are lane boundaries that are only for references. Color blocks highlighted are different segments of lanes. 
The colors of the blocks come from green and blue.
'''
# Attribute mapping   ???????
attribute_mapping = {
    0: 'unknown',
    1: 'red light',
    2: 'green light',
    3: 'yellow light',
    4: 'go_straight',
    5: 'turn_left',
    6: 'turn_right',
    7: 'no_left_turn',
    8: 'no_right_turn',
    9: 'u_turn',
    10: 'no_u_turn',
    11: 'slight_left',
    12: 'slight_right'
}

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to call GPT-4 API
def ask_GPT4(prompt, image_base64):
    response = openai.ChatCompletion.create(
        # model="gpt-4-vision-preview",
        model='gpt-4o',
        messages=[
                  {"role": "system", "content": [
                        {"type": "text", "text": system_intel},
                    ]},
                  {"role": "user", "content":[{
                        "type":"text",
                        "text":prompt
                        },
                        {
                        "type":"image_url",
                        "image_url":{
                            "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                        ]
                    }
                    ],
        max_tokens=100,
        temperature=0
    )
    return response['choices'][0]['message']['content']

# Read samples from a JSON file
# with open('/DATA_EDS2/wanggl/datasets/complete_infotest719.json', 'r') as file:
#     samples = json.load(file)

# work_path = "/DATA_EDS2/wanggl/datasets/BEV_pairwise_polygon2"
work_path = "/DATA_EDS2/wanggl/datasets/BEV_pairwise_polygon_complete_downsample"
# result_txt_path = '/DATA_EDS2/zhangzz2401/zhangzz2401/OpenLane-V2-master/mapless/ego_para_area_result.txt'
# scenes = os.listdir(work_path)
# print(scenes)
def parse_result(result_txt_path):
    with open(result_txt_path, 'r') as f:
        data = f.read()
        data = data.split('\n')
    scenes = set()
    result = {}
    for line in data:
        line = line.split(' ')
        scene = line[0]
        timestamp = line[1].split('-')[0]
        lane_id = line[1].split('-')[2]
        if scene not in scenes:
            result[scene] = {}
            timestamps = set()
        scenes.add(scene)
        if timestamp not in timestamps:
            result[scene][timestamp] = []
        timestamps.add(timestamp)
        if line[2] == "1":
            result[scene][timestamp].append(lane_id)

    return result

# result_dict = parse_result(result_txt_path)

results = ""

#   # results = {}
#   # failed_requests = []

for scene in range(10000, 10060):
    print('------------------------')
    print(f'Processing scene {scene}...')
    print('------------------------')
    scene = str(scene)
    scene_path = os.path.join(work_path, scene)
    imgs = os.listdir(scene_path)
    if len(imgs) == 0:
        line = scene + '\n'
        print(line)
        results += line
    for img in imgs:

# for scene in result_dict:
#     for timestamp in result_dict[scene]:
#         if len(result_dict[scene][timestamp]) >= 2:
#             for index1, idx1 in enumerate(result_dict[scene][timestamp]):
#                 for index2, idx2 in enumerate(result_dict[scene][timestamp]):
#                     if index1 < index2:
        try:
            img_name = img.split('.png')[0]
            img_path = os.path.join(scene_path, img)
            # img_name = timestamp + f'-ls-{idx1}-{idx2}'
            # img_path = os.path.join(work_path, scene, img_name) + '.png'
            print('Encoding image...')
            base64_image = encode_image(img_path)
            print('Image encoded.')
            prompt = f'''
You are an expert in determining positional relationships of lane segments in the image. Is the green segment on the left of the blue segment, or on the right? Please reply in a brief sentence.
        '''
#             prompt = f'''
# In the provided photograph, is the green patch on the left of the blue patch, or on the right?
#         '''
        #     prompt = '''
        #      In the provided photograph, what positional relationship is between the green patch and the blue patch? If the green patch is on the left of the blue patch, answer "1". If the green patch is on the right of the blue patch, answer "2". If you are not sure, answer "0".
        # '''
            # Call GPT-4 API
            print('Calling GPT-4 API...')
            result = ask_GPT4(prompt, base64_image)
            print('API call successful.')
            print(result)
            if "left" in result and "right" not in result:
                label = '1'
            elif "right" in result and "left" not in result:
                label = '2'
            else:
                label = '0'

            # if result == "left":
            #     label = "1"
            # elif result == "right":
            #     label = "2"
            # else:
            #     label = "0"
            line = scene + " " + img_name + " " + label + "\n"
            # line = scene + " " + img_name + " " + result + "\n"
            print(line)
            results += line
            # results[img_name] = result
        except Exception as e:
            print(f'Error processing scene {scene}: {e}')
            # failed_requests.append((scene, str(e)))
            print('Moving to next sample...')
            time.sleep(2)  # Optionally add a delay to avoid rate limiting

with open("ego_para_lr_result.txt", "w") as f:
    f.write(results)


# if __name__ == '__main__':
#     try:
#         print('Encoding image...')
#         base64_image = encode_image("/DATA_EDS2/wanggl/datasets/BEV_pairwise_polygon2/10002/315971495049927216-ls-0-1.png")
#         print('Image encoded.')
# #         prompt = f'''
# # In the provided photograph, is the green patch on the left of the blue patch, or on the right?
# #         '''
#         # prompt = '''
#         #      In the provided photograph, what positional relationship is between the green patch and the blue patch? If the green patch is on the left of the blue patch, answer 1. If the green patch is on the right of the blue patch, answer 2. If you are not sure, answer 0.".
#         # '''
# #         prompt = f'''
# # You are an expert in determining positional relationships of lane segments in the image. Let's determine if the green patch is on the left of the blue patch.
# #         '''
#         prompt = f'''
# You are an expert in determining positional relationships of lane segments in the image. Is the green patch on the left of the blue patch, or on the right?
#         '''
#         # Call GPT-4 API
#         print('Calling GPT-4 API...')
#         result = ask_GPT4(prompt, base64_image)
#         print('API call successful.')
#         print(result)
#         # if "left" in result and "right" not in result:
#         #     print(1)
#         # elif "right" in result and "left" not in result:
#         #     print(2)
#         # else:
#         #     print(0)
#     except Exception as e:
#         # print(f'Error processing scene {scene}: {e}')
#         time.sleep(2)  # Optionally add a delay to avoid rate limiting



# Process each sample
# for idx, sample in enumerate(samples):
#     try:
#         print('------------------------')
#         print(f'Processing sample {idx}...')
#         print('------------------------')

#         # image_path = sample['image_name']
#         # lane_coordinates = sample['lane_coordinates']
#         # traffic_coordinates = sample['traffic_element_coordinates']
#         # attribute = attribute_mapping[sample['attribute']]  # Map attribute number to description

#         # Generate prompt
#         prompt = f'''
# Based on the provided image where the traffic element is highlighted with a green anchor box at coordinates {traffic_coordinates} with attribute '{attribute}' , and the lane line is marked with a green line flanked by two red lines at coordinates {lane_coordinates}, please analyze whether there is a topological relationship between the designated traffic element and the lane line. Use the established rules to guide your analysis and express your reasoning in DSL format.

# '''

#         # Encode the image
#         print('Encoding image...')
#         newimgpath = '/DATA_EDS2/wanggl/datasets/Opentest_719test2' + '/' + image_path
#         base64_image = encode_image(newimgpath)
#         print('Image encoded.')

#         # Call GPT-4 API
#         print('Calling GPT-4 API...')
#         result = ask_GPT4(system_intel, prompt, base64_image)
#         print('API call successful.')

#         # Save the result with the image name as key
#         image_name = os.path.basename(image_path)
#         results[image_name] = result

#     except Exception as e:
#         print(f'Error processing sample {idx}: {e}')
#         failed_requests.append((idx, str(e)))
#         print('Moving to next sample...')
#         time.sleep(2)  # Optionally add a delay to avoid rate limiting

# # Save all results to a JSON file
# with open('gptresults01.json', 'w') as outfile:
#     json.dump(results, outfile)

# # Optionally, save failed requests to review later
# with open('failed_requests01.json', 'w') as error_file:
#     json.dump(failed_requests, error_file)




###1
'''
You are an automated traffic scene analysis system designed to identify and analyze the topological relationships between traffic elements and lane lines based on visual data. You must apply the following rules to determine the topological relationships: 
1. **Initial Analysis**: Upon receiving an image and associated coordinates, your first task is to analyze the image to determine if any intersections are present and count how many there are. The result of this analysis should be output as a number, which will directly inform subsequent rules application.
2. **Decision Branching Based on Intersections**:
   - If the result from Rule 1 is '0' (no intersections), proceed with Rule 3.
   - If the result from Rule 1 is '1' (one intersection), proceed with Rule 4.
3. **Rule 3 - Direct Influence Analysis**:
   - Analyze the provided image and coordinates to determine whether traffic elements directly influence the lane lines. Output '1' if there is a direct topological relationship, otherwise output '0'.
4. **Rule 4 - Relative Position Analysis**:
   - Assess the relative position of traffic elements to the lane lines. If a traffic element, such as a traffic light, is in front of a lane line, it can influence the lane line. If it is behind, it should not influence the lane line as the vehicle has already passed. Output '1' for direct influence and '0' for no influence.
5. **Comprehensive Judgment**:
   - Integrate the logic from the above rules to make a final determination. This involves synthesizing image analysis, intersection counting, and direct influence assessment to produce a decision. Output a concise DSL that captures the decision logic based on the analysis.
These rules should be strictly followed to ensure accuracy and consistency in the traffic scene analysis. Your outputs must be clear, directly reflecting the logical determinations made based on the visual data and the predefined rules.
'''
###2
'''
You are an automated traffic scene analysis system designed to identify and analyze the topological relationships between traffic elements and lane lines based on visual data. You must apply the following rules to determine the topological relationships:

1. **Initial Analysis**:
   - Analyze the image to determine if any intersections are present and count how many there are. This step forms the basis for further decision-making.
   - Example DSL for this step:
     ```dsl
     count(Intersection, lambda x: intersection(x, image))
     ```

2. **Decision Branching Based on Intersections**:
   - Depending on the number of intersections identified, different rules are applied:
     - If no intersections (output '0'):
       ```dsl
       apply_no_intersection_rules(traffic_elements, lane_positions)
       ```
     - If one intersection (output '1'):
       ```dsl
       apply_single_intersection_rules(traffic_elements, lane_positions)
       ```

3. **Rule Applications**:
   - **No Intersection Rules (Rule 3)**:
     - Assess whether traffic elements directly influence the lane lines based on their proximity and alignment.
     - Example DSL:
       ```dsl
       directly_affects(iota(Object, lambda x: traffic_element(x) and near(x, iota(Object, lambda y: lane_line(y)))))
       ```
   - **Single Intersection Rules (Rule 4)**:
     - Assess the relative position and direct influence of traffic elements when an intersection is involved.
     - Example DSL:
       ```dsl
       if not behind(iota(Object, lambda x: lane_line(x)), iota(Object, lambda y: intersection(y))):
           directly_affects(iota(Object, lambda x: traffic_element(x)), iota(Object, lambda y: lane_line(y)))
       else:
           output "0"
       ```

4. **Comprehensive Judgment**:
   - Integrate the logic from the above rules to make a final determination, synthesizing the analysis results into a decision.
   - Example DSL:
     ```dsl
     evaluate_decision_based_on_rules(traffic_element_type, lane_position, intersection_count)
     ```

These rules should be strictly followed to ensure accuracy and consistency in the traffic scene analysis. Your outputs must be clear, directly reflecting the logical determinations made based on the visual data and the predefined rules.
'''
###3
'''
You are an automated traffic scene analysis system designed to identify and analyze the topological relationships between traffic elements and lane lines based on visual data. Apply the following sequential rules and provide outputs at each step to determine the topological relationships:

1. **Intersection Analysis**:
   - Analyze the image to determine if any intersections are present and how many. Output the number of intersections found, e.g., "Found 1 intersection."

2. **Decision Branching Based on Intersections**:
   - Depending on the number of intersections:
     - If 0 intersections are found, output "No intersections found, applying Rule 3."
     - If 1 intersection is found, output "One intersection found, applying Rule 4."

3. **Rule 3 - Direct Influence Analysis** (applied if no intersections):
   - Analyze whether traffic elements directly influence the lane lines based on their `attribute`:
     - If `attribute` is 'left_turn' or 'no_left_turn', analyze if it affects only the leftmost lane.
     - If `attribute` is 'right_turn' or 'no_right_turn', analyze if it affects only the rightmost lane.
   - Output '1' if there is a direct topological relationship, otherwise '0'. Include a message, e.g., "Direct influence analysis based on attribute: 1 (exists) / 0 (does not exist)."

4. **Rule 4 - Relative Position Analysis** (applied if one intersection):
   - Assess the relative position of traffic elements to the lane lines and determine influence based on `attribute`:
     - For 'left_turn' or 'no_left_turn', ensure it is influencing the leftmost lane.
     - For 'right_turn' or 'no_right_turn', ensure it is influencing the rightmost lane.
   - Output '1' for correct influence and '0' for no or incorrect influence. Include a message, e.g., "Relative position analysis based on attribute: 1 (exists) / 0 (does not exist)."

5. **Comprehensive Judgment**:
   - Combine the results from the above analyses and output a final decision, e.g., "Final decision based on combined analysis: 1 / 0."
   - Provide a DSL statement that captures the decision logic.
'''


###prompt
'''
Based on the provided image where the traffic element is highlighted with a green anchor box at coordinates {traffic_coordinates} with attribute '{attribute}' , and the lane line is marked with a green line flanked by two red lines at coordinates {lane_coordinates}, please analyze whether there is a topological relationship between the designated traffic element and the lane line. Use the established rules to guide your analysis and express your reasoning in DSL format.

Output:
- '1' if there is a topological relationship according to the rules, or '0' if there is none.
- A DSL statement that describes the logic used to reach this conclusion, such as:
  ```dsl
  directly_affects(iota(Object, lambda x: traffic_element(x, {traffic_coordinates}) and on(x, iota(Object, lambda y: lane_line(y, {lane_coordinates})))), iota(Object, lambda y: lane_line(y, {lane_coordinates})))
'''