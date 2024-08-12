import setGPU
import os
import csv
import pickle
import re
import openai
from sentence_transformers import SentenceTransformer
from os import path as osp
from tqdm import tqdm
import shutil
import argparse
from safebench.util.scenic_utils import ScenicSimulator
# no need for faiss currently
# import faiss
os.environ["OPENAI_API_KEY"] = 'sk-proj-xxx'

parser = argparse.ArgumentParser(description="Set up configurations for your script.")
# Define the arguments with default values
parser.add_argument('--port_ip', type=int, default=2000, help='Port IP address (default: 2000)')
parser.add_argument('--topk', type=int, default=1, help='Top K value (default: 2)')
parser.add_argument('--model', type=str, default='gpt-4o', help="Model name (default: 'gpt-4o')")
# Parse the arguments
args = parser.parse_args()

# Access the values
port_ip = args.port_ip
topk = args.topk
gpt_model = args.model

client = openai.OpenAI()
local_path = osp.abspath(osp.dirname(osp.dirname(osp.realpath(__file__))))

def load(file_path):
    with open(file_path, 'r') as file:
        return file.read()

extraction_prompt = load(osp.join(local_path, 'retrieve', 'prompts', 'extraction.txt'))
behavior_prompt = load(osp.join(local_path, 'retrieve', 'prompts', 'behavior.txt'))
geometry_prompt = load(osp.join(local_path, 'retrieve', 'prompts', 'geometry.txt'))
spawn_prompt = load(osp.join(local_path, 'retrieve', 'prompts', 'spawn.txt'))
scenario_descriptions = load(osp.join(local_path, 'retrieve', 'scenario_descriptions.txt')).split('\n')

model = SentenceTransformer('sentence-transformers/sentence-t5-large', device='cuda')

with open(osp.join(local_path, 'retrieve/database_v1.pkl'), 'rb') as file:
    database = pickle.load(file)

behavior_descriptions = database['behavior']['description']
geometry_descriptions = database['geometry']['description']
spawn_descriptions = database['spawn']['description']
behavior_snippets = database['behavior']['snippet']
geometry_snippets = database['geometry']['snippet']
spawn_snippets = database['spawn']['snippet']

behavior_embeddings = model.encode(behavior_descriptions, device='cuda', convert_to_tensor=True)
geometry_embeddings = model.encode(geometry_descriptions, device='cuda', convert_to_tensor=True)
spawn_embeddings = model.encode(spawn_descriptions, device='cuda', convert_to_tensor=True)

head = '''param map = localPath(f'../maps/{Town}.xodr') 
param carla_map = Town
model scenic.simulators.carla.model
EGO_MODEL = "vehicle.lincoln.mkz_2017"
'''

def retrieve_topk(descriptions, snippets, embeddings, current_description):
    current_embedding = model.encode([current_description], device='cuda', convert_to_tensor=True)
    scores = (current_embedding @ embeddings.T).squeeze(0)
    top_indices = scores.topk(k=topk).indices
    top_descriptions = [descriptions[i] for i in top_indices]
    top_snippets = [snippets[i] for i in top_indices]
    
    return top_descriptions, top_snippets

def extract_scenic_code(text):
    pattern = r"```scenic(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    else:
        import pdb;pdb.set_trace()

def generate_code_snippet(category_prompt, descriptions, snippets, current_description):
    """
    Generates a code snippet using the OpenAI API based on given descriptions and snippets.

    Args:
    - model (str): The model identifier.
    - category_prompt (str): The prompt template for the specific category.
    - descriptions (list): List of top descriptions.
    - snippets (list): List of top snippets corresponding to the descriptions.
    - current_description (str): The current scenario description for which the snippet is generated.

    Returns:
    - str: The generated code snippet.
    """
    system_prompt ='''Your goal is to assist me in writing snippets using Scenic 2.1 for CARLA simulation. Scenic is a domain-specific probabilistic programming language designed for modeling environments in cyber-physical systems like robots and autonomous vehicles. Please adhere strictly to the Scenic 2.1 API, avoiding the use of any non-existent APIs or the Python random package.'''

    try:
        if topk == 1:
            return snippets[0]
            
        content = '\n'
        for j in range(topk):
            content += f'Description: {descriptions[j]}\nSnippet:\n```scenic\n{snippets[j]}```\n'
            
        response = client.chat.completions.create(
            model=gpt_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": category_prompt.format(
                    content=content,
                    current_description=current_description
                )}
            ]
        )
        # Extract the first completion from the response
        generated_code = response.choices[0].message.content
        return extract_scenic_code(generated_code)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def save_scenic_code(scenic_code, q):
    # Construct the file path dynamically using the scenario index
    file_path = osp.join(local_path, f'safebench/scenario/scenario_data/scenic_data/dynamic_scenario/dynamic_{q}.scenic')
    backup_path = osp.join(local_path, f'safebench/scenario/scenario_data/scenic_data/dynamic_scenario/dynamic_{q}.txt')

    # Open the file in write mode ('w') which will create the file if it doesn't exist or overwrite it if it does
    with open(file_path, 'w') as file:
        file.write(scenic_code)

    extra_params = {'port': port_ip, 'traffic_manager_port': port_ip + 6000}

    try:
        print("Checking if the Scenic code is compilable...")
        # Simulate the initialization of the Scenic simulator to check if the script is compilable
        ScenicSimulator(file_path, extra_params)
        print(f"Scenic code saved and verified as compilable at {file_path}")
    except Exception as e:
        import pdb;pdb.set_trace()
        # If an error occurs, log the error and move the scenic file to a .txt for further inspection
        print(f"Failure in compiling Scenic code with ID {q}: {str(e)}")
        shutil.move(file_path, backup_path)  # Move the problematic Scenic file to a .txt extension
        print(f"Moved problematic Scenic file to {backup_path}")

log_file_path = osp.join(local_path, 'safebench', 'scenario', 'scenario_data', 'scenic_data', 'dynamic_scenario', 'dynamic_log.csv')
with open(log_file_path, mode='w', newline='') as file:
    log_writer = csv.writer(file)
    log_writer.writerow(['Scenario', 'AdvObject', 'Behavior Description', 'Behavior Snippet', 'Geometry Description', 'Geometry Snippet', 'Spawn Description', 'Spawn Snippet', 'Success'])

    for q, current_scenario in tqdm(enumerate(scenario_descriptions)):
        response = client.chat.completions.create(
            model=gpt_model,
			messages=[
			{"role": "system", "content": "You are a helpful assistant."},
			{"role": "user", "content": extraction_prompt.format(scenario=current_scenario)},
			]
        ).choices[0].message.content
        match = re.search(r"Object Type:(.*?)Behavior:(.*?)Geometry:(.*?)Spawn Position:(.*)", response, re.DOTALL)
        # try:
        current_adv_object, current_behavior, current_geometry, current_spawn = [s.strip() for s in match.groups()]
        top_behavior_descriptions, top_behavior_snippets = retrieve_topk(behavior_descriptions, behavior_snippets, behavior_embeddings, current_behavior)
        top_geometry_descriptions, top_geometry_snippets = retrieve_topk(geometry_descriptions, geometry_snippets, geometry_embeddings, current_geometry)
        top_spawn_descriptions, top_spawn_snippets = retrieve_topk(spawn_descriptions, spawn_snippets, spawn_embeddings, current_spawn)
        
        generated_behavior_code = generate_code_snippet(
            behavior_prompt, 
            top_behavior_descriptions, top_behavior_snippets, current_behavior
        )

        generated_geometry_code = generate_code_snippet(
            geometry_prompt,
            top_geometry_descriptions, top_geometry_snippets, current_geometry
        )
        
        generated_spawn_code = generate_code_snippet(
            spawn_prompt,
            top_spawn_descriptions, top_spawn_snippets, current_spawn
        )
        
        log_writer.writerow([current_scenario, current_adv_object, current_behavior, generated_behavior_code, current_geometry, generated_geometry_code, current_spawn, generated_spawn_code, 1])

        Town, generated_geometry_code = generated_geometry_code.split('\n', 1)
        scenic_code = '\n'.join([f"'''{current_scenario}'''", Town, head, generated_behavior_code, generated_geometry_code, generated_spawn_code.format(AdvObject = current_adv_object)])
        save_scenic_code(scenic_code, q)

        # except:
        #     log_writer.writerow([current_scenario, '', '', '', '', '', '', '', 0])
        #     print("Failure for scenario:", current_scenario)
