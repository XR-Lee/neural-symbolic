from openai import OpenAI
import os
import json

os.environ['HTTP_PROXY'] = 'http://10.0.0.15:11452'
os.environ['HTTPS_PROXY'] = 'http://10.0.0.15:11452'

client = OpenAI(api_key="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")

def check_in_intersection(search_lane_id):
    print("check_in_intersection called")
    print(search_lane_id)
    intersection = ""
    with open(file_path, 'r') as file:
        data = json.load(file)
    #print(data)

    for info in data['information']:
        if  info['lane_id'] == search_lane_id:
            intersection = info['intersection']
            print(intersection)
            break
    return intersection

def check_connection(search_lane_id):
    print("check_connection called")

    with open(file_path, 'r') as file:
        data = json.load(file)

    connection = ""
    for info in data['information']:
        if search_lane_id == info['lane_id']:
                connection = info['connection']
    search_lane_id = connection
    print(search_lane_id)
    return search_lane_id


def traffic_elements(timestap):
    print("traffic_elements called")

    with open(file_path, 'r') as file:
        data = json.load(file)

    attribute_list = []
    for element in data['traffic_element']:
        attribute_list.append(element['attribute'])
    print(attribute_list)

    attribute_mapping = {
    0: 'unknown traffic light',
    1: 'red traffic light',
    2: 'green traffic light',
    3: 'yellow traffic light',
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

    category_list = []
    category_list = [attribute_mapping[attribute] for attribute in attribute_list]
    print(category_list)
    return category_list

def check_location(search_lane_id):
    print("check_location called")

    with open(file_path, 'r') as file:
        data = json.load(file)

    location = ""
    for info in data['information']:
         if search_lane_id == info['lane_id']:
            location = info['location']
    print(location)
    return location


tools=[
        {
            "type": "function",
            "function": {
                "name": "check_in_intersection",
                "strict": True,
                "description": "Check if the given lane is at an intersection, and if so return 0, 1 else. Call this whenever a new lane's ID is given.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_lane_id": {
                            "type": "string",
                            "description": "The marked lane's ID"
                        }
                    },
                    "required": ["search_lane_id"],
                    "additionalProperties": False,
                }
               
            }
        },
        {
            "type": "function",
            "function": {
                "name": "check_connection",
                "strict": True,
                "description": "Check the lane's ID which is adjcent to the given lane.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_lane_id": {
                            "type": "string",
                            "description": "The marked lane's ID"
                            }
                    },
                    "required": ["search_lane_id"],
                    "additionalProperties": False,
                }

            }
        },
        {
            "type": "function",
            "function": {
                "name": "traffic_elements",
                "description": "Identify all the traffic elements in front of the vehicle and return a list composed of these traffic elements.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "timestap": {
                            "type": "string",
                            "description": "The timestap of the current scene"
                            }
                    },
                    "required": ["timestap"],
                    "additionalProperties": False,
                }

            }
        },
        {
            "type": "function",
            "function": {
                "name": "check_location",
                "description": "Check the relative position of the given lane in relation to other lanes within the overall Bird's Eye View (BEV) map.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_lane_id": {
                            "type": "string",
                            "description": "The marked lane's ID"
                            }
                    },
                    "required": ["search_lane_id"],
                    "additionalProperties": False,
                }

            }
        }

    ]

available_tools = {
    "check_in_intersection": check_in_intersection,
    "check_connection": check_connection,
    "traffic_elements": traffic_elements,
    "check_location": check_location
}

def parse_file(filename, index):
    with open(filename, 'r') as file:
        lines =file.readlines()

        line = lines[index]
        parts = line.strip().split()
        sample_id = parts[0]

        part = parts[1].split('-ls-')
        timestamp = part[0]
        lane_id = part[1]
        intersection = int(parts[-1])
    return sample_id, timestamp, lane_id, intersection

input_filename = '/DATA_EDS2/zhangzz2401/zhangzz2401/OpenLane-V2-master/mapless/ego_para_area_result.txt'
output_filename = '/DATA_EDS2/zhangzz2401/zhangzz2401/OpenLane-V2-master/mapless/gptcall_loop_result_0912.txt'

with open(input_filename, 'r') as file:
    lines =file.readlines()
for index in range(len(lines)):
    data = parse_file(input_filename, index)
    sample_id = data[0]
    timestamp = data[1]
    lane_id = data[2]
    intersection = data[3]

    file_path = '/DATA_EDS2/zhangzz2401/zhangzz2401/OpenLane-V2-master/output/' + sample_id + '_' + timestamp + '.json'
    print(file_path, lane_id)

    with open(file_path,'r') as file:
        json_data = json.load(file)
        if json_data['traffic_element'] == []:
            continue

    if intersection == 0:
        continue
    else:
        search_lane_id = lane_id
        
        messages = [{"role": "user", "content": f"You are now an expert of question answering for lanes and traffic \
                    elements topological relationship reasoning. When provided with the current \
                    driving scene {timestamp} and an ID {search_lane_id} for a lane, first assess whether the lane \
                    is at an intersection. If the function check_in_intersection returns a value of 0, \
                    it signifies that the lane is at an intersection, and no other function is necessary. \
                    All the function calling will terminate and for this lane we return a list of [0] as the result.\
                    If the function check_in_intersection returns a value of 1, it indicates that the lane is not \
                    at an intersection. \
                    Next, we need to determine if there is a relationship between the traffic elements \
                    in the current scene and the given lane. We use the 'traffic_elements' function to obtain a list of all detected traffic elements \
                    including traffic lights and traffic signs in the current scene. When the category of the traffic element in the list is traffic\
                    light including 'green traffic light','red traffic light' or 'yellow traffic light', we can consider \
                    that this traffic element is related to the lane, and we return a value of 1. \
                    If the traffic element's category is traffic sign like 'left turn', 'no left turn', u-turn', 'no U-turn' or 'slight left', \
                    then the check_location function needs to be called to determine the relative \
                    position of the given lane within the overall Bird's Eye View (BEV) map.These signs are only topologically related to the lane\
                    on the left side.So if the result of check_location function is left, it means the given lane is on the left side, and we return\
                    a value of 1, otherwise we return 0.\
                    When the traffic element's category in the list is traffic sign including 'right turn' 'no right turn' or 'slight right', \
                    the check_location function should also be called. They are only related to the lane\
                    on the right side and not related to the lane on the left side. So if the result of check_location function is left,\
                    it means the given lane is on the left side, and we return 0. If the result is right, we return 1 which means the given lane\
                    is related to the traffic sign.\
                    If the traffic sign is 'go straight', the check_location function should be called. If the lane is on the left or in the middle,\
                    there is a topological relationship between the straight sign and the lane, and the return value is 1. \
                    For the lane on the right, there is no topological relationship, hence the return value is 0. \
                    Additionally, if the result of the check_location function is 'single', it indicates that the given lane is a one-way street.\
                    We consider this lane has topological relationship with all traffic elements except for the 'unknown' traffic elements.\
                    Any traffic elements that do not fall into these categories are considered unrelated to the lane, and the output should be 0. \
                    Please return a list of topological results indicating whether each traffic element in the provided list is relevant or irrelevant\
                    to the lane, for example [0 , 1, 1, 0 ]. It's crucial that the output topological results must correspond one-to-one with the types of traffic elements.\
                    After determining the topological relationship between the traffic elements and the lane, necessitate a further examination of the \
                    lane ID that are connected to the given lane. For each newly identified lane, continue this process iteratively \
                    until all lanes that meet the criteria have been identified.\
                    Return the summary of the results for the given lane in requested format"}]

        iters = 0
        while iters < 10 :
            iters+=1
            response = client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
            
            response_msg = response.choices[0].message
            messages.append(response_msg)
            print("----------------------")
            print(response_msg.content)
            print("----------------------")

            #print(response_msg.content) if response_msg.content else print(response_msg.tool_calls)

            finish_reason = response.choices[0].finish_reason
            if finish_reason == "stop":
                print("stopped")
                break

            tool_calls = response_msg.tool_calls
            if tool_calls:
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_to_call = available_tools[function_name]
                    function_args = json.loads(tool_call.function.arguments)
                    function_response = function_to_call(**function_args)

                    messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": str(function_response),
                        }
                    )
        
    print("result finalize")
    print(response_msg.content)

    with open(output_filename, 'a', encoding='utf-8') as file:
        file.write(f"{sample_id}-{timestamp}:{response_msg.content}\n")
        file.write(f"------------------------------------------------------------------------------------\n")
    print(f"{sample_id},{timestamp}内容已经写入")


