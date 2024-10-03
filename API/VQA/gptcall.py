import openai
from openai import OpenAI
import os
import base64
import json
from IPython.display import Markdown, display
import time

os.environ['HTTP_PROXY'] = 'http://10.0.0.15:11452'
os.environ['HTTPS_PROXY'] = 'http://10.0.0.15:11452'

# openai.api_key = os.getenv("OPENAI_API_KEY")
# openai.api_key= "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"

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

def rule1(category):
    print("rule1 called")
    if category == "unknown":
        return 0
    else:
        return 1

client = OpenAI(api_key="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")
assistant = client.beta.assistants.create(
    instructions="You are an ego driving bot. Use the provided functions to answer questions.",
    # model="gpt-4-vision-preview",
    model="gpt-4o",
    tools=[
        {
            "type": "function",
            "function": {
                "name": "rule1",
                "description": "Check if the attribute of the traffic element is unknown with its given attribute, and if so return 0, 1 else",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "The attribute of the traffic element"
                        }
                    },
                    "required": ["category"]
                }
            }
        }
    ]
)


with open('/DATA_EDS2/zousz2407/topo/complete_infotest7192.json', 'r') as file:
    samples = json.load(file)

for idx, sample in enumerate(samples):
    if idx == 10:
        break
    try:
        print('------------------------')
        print(f'Processing sample {idx}...')
        print('------------------------')

        image_path = sample['image_name']
        area_points = sample['area']
        transformed_lane_lines = sample['lane_coordinates']
        traffic_coordinates = sample['traffic_element_coordinates']
        attribute = attribute_mapping[sample['attribute']]

        # print('Encoding image...')
        # newimgpath = '/DATA_EDS2/wanggl/DATA/data719' + '/' + image_path
        # base64_image = encode_image(newimgpath)
        # print('Image encoded.')

        thread = client.beta.threads.create()
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=f"Is the attribute of the traffic element 'unknown'? With its given attribute {attribute}",
        )


        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id=assistant.id,
        )
        
        if run.status == 'completed':
            messages = client.beta.threads.messages.list(
                thread_id=thread.id
            )
            print(messages)
        else:
            print(run.status)
        
        # Define the list to store tool outputs
        tool_outputs = []
        
        # Loop through each tool in the required action section
        for tool in run.required_action.submit_tool_outputs.tool_calls:
            if tool.function.name == "rule1":
                args = json.loads(tool.function.arguments)
                answer = rule1(
                    category=args.get("category")
                )
                print(args.get("category"), answer)
                tool_outputs.append({
                    "tool_call_id": tool.id,
                    "output": str(answer)
                })
            #   elif tool.function.name == "get_rain_probability":
            #     tool_outputs.append({
            #       "tool_call_id": tool.id,
            #       "output": "0.06"
            #     })
        
        # Submit all tool outputs at once after collecting them in a list
        if tool_outputs:
            try:
                run = client.beta.threads.runs.submit_tool_outputs_and_poll(
                thread_id=thread.id,
                run_id=run.id,
                tool_outputs=tool_outputs
                )
                print("Tool outputs submitted successfully.")
            except Exception as e:
                print("Failed to submit tool outputs:", e)
        else:
            print("No tool outputs to submit.")
        
        if run.status == 'completed':
            messages = client.beta.threads.messages.list(
                thread_id=thread.id
            )
            print(messages)
        else:
            print(run.status)

    except Exception as e:
        print(f'Error processing sample {idx}: {e}')
        # failed_requests.append((idx, str(e)))
        print('Moving to next sample...')
        time.sleep(2)  # Optionally add a delay to avoid rate limiting
