from mcp.server import FastMCP
import re
from openai import OpenAI
import os
from dotenv import load_dotenv
import numpy as np
import json
from tqdm import tqdm
import copy
import sys
from io import StringIO
import traceback
import time
from datetime import datetime

load_dotenv()  # load environment variables from .env
XAI_API_KEY = os.getenv("XAI_API_KEY")

# Initialize FastMCP
mcp = FastMCP("ns-mcp")

def write_to_log(message: str, log_file):
    """Write message to both log file and stdout."""
    print(message)  # Print to console
    log_file.write(message + "\n")  # Write to log file
    log_file.flush()  # Ensure it's written immediately

@mcp.tool()
async def generate_py_codes() -> str:
    """Generate Python code for lane connectivity analysis."""
    client = OpenAI(
        api_key=XAI_API_KEY,
        base_url="https://api.x.ai/v1",
    )

    try:
        completion = client.chat.completions.create(
            model="grok-3-beta",
            messages=[
                {"role": "system", "content": "Generate a Python function that properly handles numpy arrays and calculates distances between lane points."},
                {"role": "user", "content": """Write a condition function to determine if two lanes are connected. The function must:
1. Be named 'condition'
2. Take parameters (ref_centerline, cmp_centerline, ref_leftline, ref_rightline, cmp_leftline, cmp_rightline)
3. Handle inputs as numpy arrays where each array contains points with shape (N, 2) for x,y coordinates
4. Calculate the Euclidean distance between the last point of ref_centerline and first point of cmp_centerline using numpy
5. Return True if the distance is LESS than a threshold (e.g., 5.0 meters), False otherwise
6. Use proper numpy operations (np.linalg.norm) for distance calculation
7. Include error handling for empty arrays

Example calculation (but write the complete function):
distance = np.linalg.norm(ref_centerline[-1] - cmp_centerline[0])
return distance < threshold

Make sure to handle numpy arrays correctly and avoid direct boolean operations on arrays."""}
            ],
        )

        gpt_response = completion.choices[0].message.content
        print("Raw response:", gpt_response)

        code_block = re.search(r'```python(.*?)```', gpt_response, re.DOTALL)
        if code_block:
            python_code = code_block.group(1).strip()
        else:
            print("No Python code block found.")
            python_code = ""

        base_code = '''
# Utility function to transform coordinates
def transform_coord(line):
    # Convert to numpy array if it's a list
    if not isinstance(line, np.ndarray):
        line = np.array(line, dtype=np.float32)
    # Now transform the coordinates
    return np.array([[-point[1], point[0]] for point in line], dtype=np.float32)

print("Loading JSON data...")
with open(json_file_path, 'r') as f:
    json_data = json.load(f)

print("Processing predictions...")
predictions = json_data['predictions']
lane_segments = predictions['lane_segment']
topll_matrix = predictions['topology_lsls']
mask_ids = []

print(f"Processing {len(lane_segments)} lane segments...")
for ref_lane in tqdm(lane_segments, desc='Processing lanes'):
    mask_id = []
    try:
        ref_centerline = transform_coord(ref_lane['centerline'])
        ref_leftline = transform_coord(ref_lane['left_laneline'])
        ref_rightline = transform_coord(ref_lane['right_laneline'])
        
        for index, cmp_lane in enumerate(lane_segments):
            try:
                cmp_centerline = transform_coord(cmp_lane['centerline'])
                cmp_leftline = transform_coord(cmp_lane['left_laneline'])
                cmp_rightline = transform_coord(cmp_lane['right_laneline'])
                
                if condition(ref_centerline, cmp_centerline, ref_leftline, ref_rightline, cmp_leftline, cmp_rightline):
                    mask_id.append(index)
                    pair_cnt += 1
            except Exception as e:
                print(f"Error processing comparison lane {index}: {str(e)}")
                continue
        
        max_pair = max(max_pair, len(mask_id))
        mask_ids.append(mask_id)
    except Exception as e:
        print(f"Error processing reference lane: {str(e)}")
        mask_ids.append([])
        continue

print("Updating topology matrix...")
for i in range(len(mask_ids)):
    for j in range(len(mask_ids)):
        if j in mask_ids[i]:
            topll_matrix[i][j] = 1
        else:
            topll_matrix[i][j] = 0

print("Final topology matrix shape:", np.array(topll_matrix).shape)
print("Number of connections found:", pair_cnt)
print("Maximum connections per lane:", max_pair)
'''

        final_codes = python_code + "\n" + base_code
        return final_codes

    except Exception as e:
        print(f"Error generating Python code: {e}")
        return f"Error occurred: {str(e)}"


@mcp.tool()
async def execute_generated_code(code: str, json_file_path: str = None) -> dict:
    """Execute the generated Python code in a controlled environment."""
    start_time = time.time()
    
    # Create log directory if it doesn't exist
    log_dir = "execution_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"execution_log_{timestamp}.txt")
    
    with open(log_path, 'w') as log_file:
        # Store original stdout for actual console output
        actual_stdout = sys.stdout
        
        write_to_log(f"\n=== Starting Execution at {timestamp} ===", log_file)
        write_to_log(f"Processing file: {json_file_path}", log_file)
        
        stdout_buffer = StringIO()
        stderr_buffer = StringIO()
        
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        result = {
            "success": False,
            "output": "",
            "error": "",
            "execution_time": 0,
            "log_file": log_path  # Include log file path in result
        }

        if not json_file_path:
            result["error"] = "Missing required parameter: json_file_path"
            write_to_log("Error: Missing JSON file path", log_file)
            return result

        if not os.path.exists(json_file_path):
            result["error"] = f"JSON file not found: {json_file_path}"
            write_to_log(f"Error: JSON file not found: {json_file_path}", log_file)
            return result
        
        try:
            write_to_log("\n=== Code to Execute ===", log_file)
            write_to_log(code, log_file)
            write_to_log("\n=== Execution Output ===", log_file)
            
            sys.stdout = stdout_buffer
            sys.stderr = stderr_buffer
            
            # Initialize required variables
            init_code = """
import numpy as np
from tqdm import tqdm
import json
pair_cnt = 0
max_pair = 0
            """
            
            # Create namespace with all required variables and functions
            namespace = {
                'np': np,
                'json': json,
                'tqdm': tqdm,
                'copy': copy,
                'json_file_path': json_file_path,
                'print': lambda *args, **kwargs: write_to_log(" ".join(str(arg) for arg in args), log_file)
            }
            
            # First execute initialization code
            exec(init_code, namespace)
            
            # Load and validate JSON
            try:
                with open(json_file_path, 'r') as f:
                    json_data = json.load(f)
                    write_to_log(f"\nLoaded JSON data with keys: {list(json_data.keys())}", log_file)
                    if 'predictions' in json_data:
                        write_to_log(f"Predictions keys: {list(json_data['predictions'].keys())}", log_file)
                        if 'lane_segment' in json_data['predictions']:
                            write_to_log(f"Number of lane segments: {len(json_data['predictions']['lane_segment'])}", log_file)
                    if not isinstance(json_data, dict) or 'predictions' not in json_data:
                        raise ValueError("Invalid JSON format: missing 'predictions' key")
                    namespace['json_data'] = json_data
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON format: {str(e)}")
            except Exception as e:
                raise ValueError(f"Failed to load JSON file: {str(e)}")
            
            write_to_log("\n=== Starting Main Code Execution ===", log_file)
            exec(code, namespace)
            write_to_log("\n=== Main Code Execution Completed ===", log_file)
            
            # Get any stored variables we want to check
            topology_matrix = namespace.get('topll_matrix', None)
            if topology_matrix is not None:
                write_to_log(f"\nFinal topology matrix shape: {np.array(topology_matrix).shape}", log_file)
            
            pair_count = namespace.get('pair_cnt', 0)
            write_to_log(f"Total pairs found: {pair_count}", log_file)
            
            result["output"] = stdout_buffer.getvalue()
            result["success"] = True
            
        except Exception as e:
            error_msg = f"Execution error: {str(e)}\n{traceback.format_exc()}"
            write_to_log(f"\n=== Execution Error ===\n{error_msg}", log_file)
            result["error"] = error_msg
            result["success"] = False
        
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            if stderr_buffer.getvalue():
                write_to_log("\n=== Stderr Output ===", log_file)
                write_to_log(stderr_buffer.getvalue(), log_file)
                result["error"] += f"\nStderr: {stderr_buffer.getvalue()}"
            
            stdout_buffer.close()
            stderr_buffer.close()
            
            execution_time = time.time() - start_time
            result["execution_time"] = execution_time
            write_to_log(f"\n=== Execution completed in {execution_time:.2f} seconds ===", log_file)
            write_to_log(f"Log file saved to: {log_path}", log_file)
        
        return result

@mcp.tool()
async def generate_and_execute_code(json_file_path: str = None) -> dict:
    """Generate and execute lane connectivity analysis code."""
    try:
        print(f"Generating code for JSON file: {json_file_path}")
        generated_code = await generate_py_codes()
        
        if not generated_code or "Error occurred:" in generated_code:
            return {
                "success": False,
                "error": f"Code generation failed: {generated_code}",
                "generated_code": "",
                "execution_result": None
            }
        
        print("Executing generated code...")
        execution_result = await execute_generated_code(generated_code, json_file_path)
        
        return {
            "success": execution_result["success"],
            "generated_code": generated_code,
            "execution_result": execution_result
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error in generate_and_execute_code: {str(e)}",
            "generated_code": "",
            "execution_result": None
        }

if __name__ == "__main__":
    mcp.run(transport="stdio")