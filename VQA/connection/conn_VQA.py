import openai
import os
import argparse
import base64
import json
from IPython.display import Markdown, display
import time


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to call GPT-4 API
def ask_GPT4(prompt, image_base64):
    response = openai.ChatCompletion.create(
        model="gpt-4-vision-preview",
      # model='gpt-4o',
        messages=[
                  {"role": "system", "content": [
                        {"type": "text", "text": system_context},
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt', help='text prompt path', default='./VQA/connection/data/conn_text_prompt.txt')
    parser.add_argument('--visual', help='visual prompt path', default='./VQA/connection/data/conn_visual_prompt')
    parser.add_argument('--output', help='output result path', default='./VQA/connection/data/result.txt')
    parser.add_argument('--key', help='openai api key')
    parser.add_argument('--verbose', action='store_true', default=True)
    args = parser.parse_args()

    with open(args.txt, 'r') as f:
        txt_prompt = f.read()
    exec(txt_prompt)
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_key= args.key
    work_path = args.visual

    results = ""
    for scene in range(10000, 10150):
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
            try:
                img_name = img.split('.png')[0]
                img_path = os.path.join(scene_path, img)
                print('Encoding image...')
                base64_image = encode_image(img_path)
                print('Image encoded.')

                # Call GPT-4 API
                print('Calling GPT-4 API...')
                result = ask_GPT4(prompt, base64_image)
                print('API call successful.')
                if "Yes" in result :
                    label = '1'
                else:
                    label = '0'

                line = scene + " " + img_name + " " + label + "\n"
                if args.verbose:
                    print(result)
                    print(line)
                results += line
            except Exception as e:
                print(f'Error processing scene {scene}: {e}')
                print('Moving to next sample...')
                time.sleep(2)

    with open(args.output, "w") as f:
        f.write(results)