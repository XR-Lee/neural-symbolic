from openai import OpenAI
import os
import json
from pydantic import BaseModel


# Define the CalendarEvent model
class SiglelaneFormat(BaseModel):
    lane_id: str
    results: list[bool]

class OutputFromat(SiglelaneFormat):
    all_frames: list[SiglelaneFormat]

os.environ['HTTP_PROXY'] = 'http://10.0.0.15:11452'
os.environ['HTTPS_PROXY'] = 'http://10.0.0.15:11452'

client = OpenAI(api_key="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")

sample_id = "10002"
timestap = "315971486049927220"
search_lane_id = "20013"

'''
class Topols(BaseModel):
    timestap : str
    search_lane_id : str
    Topo_results : [str]
'''

def check_in_intersection(search_lane_id):
    print("rule2 called")
    print(search_lane_id)
    intersection = ""
    with open('/home/iix5sgh/workspace/llm/20240808.json', 'r') as file:
        data = json.load(file)
    #print(data)

    for info in data['information']:
        if  info['lane_id'] == search_lane_id:
            intersection = info['intersection']
            print(intersection)
            break
    return intersection

def check_connection(search_lane_id):
    print("rule3 called")

    with open('/home/iix5sgh/workspace/llm/20240808.json', 'r') as file:
        data = json.load(file)

    connection = ""
    for info in data['information']:
         if search_lane_id == info['lane_id']:
                connection = info['connection']
    search_lane_id = connection
    print(search_lane_id)
    return search_lane_id

def traffic_elements(timestap):
    print("rule4 called")

    with open('/home/iix5sgh/workspace/llm/20240808.json', 'r') as file:
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
                "strict": True,
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
        }

    ]

available_tools = {
    "check_in_intersection": check_in_intersection,
    "check_connection": check_connection,
    "traffic_elements": traffic_elements,
}
 
messages = [{"role": "user", "content": f"You are now an expert in lanes and traffic \
            elements topological relationship annotation. When provided with the current \
            driving scene {timestap} and an ID {search_lane_id} for a lane,first assess \
            whether this lane is at an intersection. If the function check_in_intersection \
            returns a value of 0, it signifies that the lane is at an intersection, \
            and no other function is necessary. All the function calling will terminate \
            and we should add lane_id and a list of [0] to the result. If the function \
            check_in_intersection returns a value of 1, it indicates that the lane is not \
            at an intersection. Next, we need to determine if there is a relationship \
            between the traffic elements in the current scene and the lane. We use the \
            'traffic_elements' function to obtain a list of all detected traffic elements \
            in the current scene. When the category of the traffic element is 'green \
            traffic light','red traffic light' or 'yellow traffic light', we can consider \
            that this traffic element is related to the road, and we return a value of 1. \
            Any traffic elements that do not fall into these three categories are \
            considered unrelated to the lane, and the output should be 0. Please return \
            a list of topological results indicating whether each traffic element in the \
            provided list is relevant or irrelevant to the lane, for \
            example [0 , 1, 1, 0 ]. After determining the topological relationship between \
            the traffic elements and the lane, necessitate a further examination of the \
            lane ID that are connected to the given lane. For each newly identified lane, \
            continue this process iteratively until all lanes that meet the criteria have \
            been identified. return the summery of the results for each lane_id in requested format"}]

iters=0
while iters<10 :
    iters+=1
    response = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        response_format=OutputFromat,
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
            
print(response_msg.parsed)