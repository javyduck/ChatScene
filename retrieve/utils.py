import re
from os import path as osp
import shutil
from safebench.util.scenic_utils import ScenicSimulator

def load_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def retrieve_topk(model, topk, descriptions, snippets, embeddings, current_description):
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

def generate_code_snippet(llm_model, category_prompt, descriptions, snippets, current_description, topk, use_llm):
    """
    Generates a code snippet using the OpenAI API based on given descriptions and snippets.

    Args:
    - model (str): The model identifier.
    - category_prompt (str): The prompt template for the specific category.
    - descriptions (list): List of top descriptions.
    - snippets (list): List of top snippets corresponding to the descriptions.
    - current_description (str): The current scenario description for which the snippet is generated.
    - topk (int): number of retrieval examples used
    - use_llm (bool): use llm for generating new snippets or not

    Returns:
    - str: The generated code snippet.
    """
    
    system_prompt ='''Your goal is to assist me in writing snippets using Scenic 2.1 for CARLA simulation. Scenic is a domain-specific probabilistic programming language designed for modeling environments in cyber-physical systems like robots and autonomous vehicles. Please adhere strictly to the Scenic 2.1 API, avoiding the use of any non-existent APIs or the Python random package.'''

    try:
        if not use_llm:
            ## pure retrieval
            return snippets[0]
            
        content = '\n'
        for j in range(topk):
            content += f'Description: {descriptions[j]}\nSnippet:\n```scenic\n{snippets[j]}```\n'

        messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": category_prompt.format(
                    content=content,
                    current_description=current_description
                )}
            ]
        generated_code = llm_model.generate(messages)
        return extract_scenic_code(generated_code)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def save_scenic_code(local_path, port_ip, scenic_code, q):
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
        return True
    except Exception as e:
        ### TODO: use llm for fixing the error
        # import pdb;pdb.set_trace()
        # If an error occurs, log the error and move the scenic file to a .txt for further inspection
        print(f"Failure in compiling Scenic code with ID {q}: {str(e)}")
        shutil.move(file_path, backup_path)  # Move the problematic Scenic file to a .txt extension
        print(f"Moved problematic Scenic file to {backup_path}")
        return False