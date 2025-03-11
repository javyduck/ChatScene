import setGPU
import os
import csv
import pickle
import re
from sentence_transformers import SentenceTransformer
from os import path as osp
from tqdm import tqdm
import argparse
from architecture import LLMChat
from utils import load_file, retrieve_topk, generate_code_snippet, save_scenic_code

# no need for faiss currently
# import faiss

parser = argparse.ArgumentParser(description="Set up configurations for your script.")
parser.add_argument('--port_ip', type=int, default=2000, help='Port IP address (default: 2000)')
parser.add_argument('--topk', type=int, default=3, help='Top K value (default: 3) for retrieval')
parser.add_argument('--model', type=str, default='gpt-4o', help="Model name (default: 'gpt-4o'), also support transformers model")
parser.add_argument('--use_llm', action='store_true', help='if use llm for generating new snippets')
args = parser.parse_args()

port_ip = args.port_ip
topk = args.topk
use_llm = args.use_llm

llm_model = LLMChat(args.model)
local_path = osp.abspath(osp.dirname(osp.dirname(osp.realpath(__file__))))
extraction_prompt = load_file(osp.join(local_path, 'retrieve', 'prompts', 'extraction.txt'))
behavior_prompt = load_file(osp.join(local_path, 'retrieve', 'prompts', 'behavior.txt'))
geometry_prompt = load_file(osp.join(local_path, 'retrieve', 'prompts', 'geometry.txt'))
spawn_prompt = load_file(osp.join(local_path, 'retrieve', 'prompts', 'spawn.txt'))
scenario_descriptions = load_file(osp.join(local_path, 'retrieve', 'scenario_descriptions.txt')).split('\n')
encoder = SentenceTransformer('sentence-transformers/sentence-t5-large', device='cuda')

with open(osp.join(local_path, 'retrieve/database_v1.pkl'), 'rb') as file:
    database = pickle.load(file)
behavior_descriptions = database['behavior']['description']
geometry_descriptions = database['geometry']['description']
spawn_descriptions = database['spawn']['description']
behavior_snippets = database['behavior']['snippet']
geometry_snippets = database['geometry']['snippet']
spawn_snippets = database['spawn']['snippet']

behavior_embeddings = encoder.encode(behavior_descriptions, device='cuda', convert_to_tensor=True)
geometry_embeddings = encoder.encode(geometry_descriptions, device='cuda', convert_to_tensor=True)
spawn_embeddings = encoder.encode(spawn_descriptions, device='cuda', convert_to_tensor=True)

## this is the head for scenic file, you can modified the carla map or ego model here
head = '''param map = localPath(f'../maps/{Town}.xodr') 
param carla_map = Town
model scenic.simulators.carla.model
EGO_MODEL = "vehicle.lincoln.mkz_2017"
'''

log_file_path = osp.join(local_path, 'safebench', 'scenario', 'scenario_data', 'scenic_data', 'dynamic_scenario', 'dynamic_log.csv')
with open(log_file_path, mode='w', newline='') as file:
    log_writer = csv.writer(file)
    log_writer.writerow(['Scenario', 'AdvObject', 'Behavior Description', 'Behavior Snippet', 'Geometry Description', 'Geometry Snippet', 'Spawn Description', 'Spawn Snippet', 'Success'])

    for q, current_scenario in tqdm(enumerate(scenario_descriptions)):
        messages=[
			{"role": "system", "content": "You are a helpful assistant."},
			{"role": "user", "content": extraction_prompt.format(scenario=current_scenario)},
			]
        response = llm_model.generate(messages)
        match = re.search(r"Adversarial Object:(.*?)Behavior:(.*?)Geometry:(.*?)Spawn Position:(.*)", response, re.DOTALL)
        # try:
        current_adv_object, current_behavior, current_geometry, current_spawn = [s.strip() for s in match.groups()]
        top_behavior_descriptions, top_behavior_snippets = retrieve_topk(encoder, topk, behavior_descriptions, behavior_snippets, behavior_embeddings, current_behavior)
        top_geometry_descriptions, top_geometry_snippets = retrieve_topk(encoder, topk, geometry_descriptions, geometry_snippets, geometry_embeddings, current_geometry)
        top_spawn_descriptions, top_spawn_snippets = retrieve_topk(encoder, topk, spawn_descriptions, spawn_snippets, spawn_embeddings, current_spawn)
        
        generated_behavior_code = generate_code_snippet(
            llm_model, behavior_prompt, top_behavior_descriptions, top_behavior_snippets, current_behavior, topk, use_llm
        )

        generated_geometry_code = generate_code_snippet(
            llm_model, geometry_prompt, top_geometry_descriptions, top_geometry_snippets, current_geometry, topk, use_llm
        )
        
        generated_spawn_code = generate_code_snippet(
            llm_model, spawn_prompt, top_spawn_descriptions, top_spawn_snippets, current_spawn, topk, use_llm
        )
        
        log_writer.writerow([current_scenario, current_adv_object, current_behavior, generated_behavior_code, current_geometry, generated_geometry_code, current_spawn, generated_spawn_code, 1])

        Town, generated_geometry_code = generated_geometry_code.split('\n', 1)
        scenic_code = '\n'.join([f"'''{current_scenario}'''", Town, head, generated_behavior_code, generated_geometry_code, generated_spawn_code.format(AdvObject = current_adv_object)])
        save_scenic_code(local_path, port_ip, scenic_code, q)

        # except:
        #     log_writer.writerow([current_scenario, '', '', '', '', '', '', '', 0])
        #     print("Failure for scenario:", current_scenario)
