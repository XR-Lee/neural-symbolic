import openai
import os
import base64
import json
from PIL import Image
from IPython.display import Markdown, display
import time
import io 

# Function to encode the image
def encode_image(image_path):
    with Image.open(image_path) as image_file:
        if image_file.mode == 'RGBA':
            image_file = image_file.convert('RGB')
        resized_image = image_file.resize((511,1016),Image.BILINEAR)
        
        # Save the resized image to a bytes buffer
        buffer = io.BytesIO()
        resized_image.save(buffer, format="JPEG")  # Adjust format as needed (e.g., PNG, JPEG)
        
        # Get the byte data from the buffer
        image_data = buffer.getvalue()
        return base64.b64encode(image_data).decode('utf-8')

# Function to call GPT-4 API
def ask_GPT4(prompt, image_base64):
    system_intel = '''
In the provided bird's-eye view (BEV), the red lines in the photos are lane boundaries that are only for references. Color blocks highlighted are different segments of lanes. 
The colors of the blocks come from green and blue.
'''
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



def main():
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_key= "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"
     
    work_path = "/home/iix5sgh/workspace/llm/vqa_lr_0909"
        
    results = ""

    for scene in range(10000, 10003):
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
            continue
        for img in imgs:
            try:
                img_name = img.split('.png')[0]
                img_path = os.path.join(scene_path, img)
                print('Encoding image...')
                base64_image = encode_image(img_path)
                print('Image encoded.')
                prompt = f'''
    You are an expert in determining positional relationships of lane segments in the image. Is the green segment on the left of the blue segment, or on the right? Please reply in a brief sentence.
            '''
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
    return


if __name__ == '__main__':
    main()
