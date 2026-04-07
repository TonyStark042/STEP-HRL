from openai import OpenAI
from util.inst import key_factor_prompt, local_progress_prompt_0_9, local_progress_prompt_final
import json, os
from scienceworld import ScienceWorldEnv
from concurrent.futures import ThreadPoolExecutor, as_completed
from env.scienceworld.convert_data_memory import process_obs
# from env.scienceworld.convert_data import process_obs
import threading
from tqdm import tqdm
print_lock = threading.Lock()

client = OpenAI(
    api_key="--",
    base_url="https://api.deepseek.com",
)

local_progress_prompt = """You are an AI agent responsible for updating local progress. Local progress is a short cumulative summary of what has been achieved within current subtask so far.

Inputs:
- current subtask
- previous local progress
- current action
- observation

--------------------
GLOBAL RULES: 
1. CRITICAL TOKEN RULE:
  - All object and location names MUST EXACTLY match the strings in the subtask or observation.
  - Do NOT rephrase, split, normalize, or naturalize names.

2. NO INVENTED FACTS:
  - Do NOT infer properties, conditions, or failures unless explicitly stated in the observation or subtask.

3. Task type:
  - Subtasks starting with "Locate and pick up" or "Locate and use" are Locate tasks.
  - All other subtasks are Non-Locate tasks.

--------------------
LOCATE TASKS

Output format:
<one short sentence about progress> || [Checked: loc1, loc2, ...]

Rules:
1. Checked inheritance:
  - All previously listed Checked locations MUST be retained.

2. Checked update:
  - Add the current location to Checked ONLY IF:
    1) a search-related action was performed (go, open, examine, etc.), AND
    2) the observation confirms the target is NOT present there.

3. Success rules:
  - If the target is found or picked up, the search is completed and MUST NOT continue.

4. NEW + EXCEPT OVERRIDE:
  For subtasks like "Locate and pick up a new <OBJ> except <OBJ> N":
  - Any OBJ belonging to the same name but with different numbers can be treated as a new one. (For example, "spraybottle 1" is a new one to "spraybottle 2", "apple1" is a new one to "apple 3", etc.)
  - Picking up any <OBJ> where number different from N immediately completes the subtask and don't restart it.

5. If the target is not found:
  - The sentence MUST include the word "unchecked" and imply to search unchecked locations, but MUST NOT mention or hint at any specific unchecked locations.
  - If there are door closed, the sentence MUST imply this CLOSED fact.
  - If there are door opened, the sentence MUST imply this OPENED fact.

6. Language constraints:
  - Do NOT mention locations have been checked in the sentence.
  - Do NOT repeat fixed templates such as: "Search for X continues"\"Continuing to search for X",or any near-identical sentence structures.
  - Each updated local progress sentence MUST use a different phrasing pattern than the previous one.   
  - Do NOT reuse more than 3 consecutive words from the previous local progress sentence.

--------------------
NON-LOCATE TASKS (Place / Clean / Heat / Cool / Use)

Output format:
<one short sentence about progress>
Rules:
1. Do NOT include Checked or any tags.

2. Do NOT mention the operated object unless it is explicitly mentioned in the current action.
   (Example: if the subtask is "put apple 1 in fridge", you MUST NOT mention "apple 1" unless the action mentions it.)

3. NO EXISTENCE OR LOCATION STATEMENTS:
   You MUST NOT state or imply:
   - whether an object is inside or outside any container,
   - whether an object was found or not found,
   - whether a container contains or does not contain an object,
   - whether an object is present, absent, missing, or unavailable.

4. NO PRECONDITION OR FAILURE REASONING:
   You MUST NOT explain progress using conditions such as:
   - "but X is not inside Y",
   - "because X was not found",
   - "X remains uncooled since ...",
   - any justification based on object absence or location.

5. ASSUMED AVAILABILITY:
   Assume the target object is available and operable by definition
   of the current subtask.
   Do NOT question, verify, or restate this assumption.

6. ALLOWED CONTENT ONLY:
   The output sentence may ONLY describe:
   - the execution of the intended operation, or
   - the resulting state change directly caused by the action,
   WITHOUT any reference to object availability, location, or discovery.

"""

# local_progress_prompt = local_progress_prompt_final

def llm(prompt):
    sys_prompt = local_progress_prompt
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
      model="deepseek-chat",
      messages = messages,
      temperature=0.15,
      max_tokens=1024,
      top_p=1,
      frequency_penalty=0.0,
      presence_penalty=0.0,
    #   stop=stop
    )
    return response.choices[0].message.content


def generate_local_progress(data):
    local_progresses = list()
    global_len = 0

    for idx, group in enumerate(data["group_action"]):
        subtask = data["subtask"][idx]

        local_progress = "None"
        inside_progress = [local_progress]
        
        for idx2, action in enumerate(group):
            global_idx = global_len + idx2
            
            if global_idx + 1 >= len(data['obs']):
                break
            obs = process_obs(data['obs'][global_idx + 1], mode="alfworld")
            prompt = f"""Subtask: {subtask}
    Previous local progress: {local_progress}
    Current action: {action}
    Resulting observation:\n {obs}\n
    Updated local progress:"""
            local_progress = llm(prompt)
            inside_progress.append(local_progress)
            # print(f"Subtask: {subtask}\nPrevious local progress: {inside_progress[-1]}\nCurrent action: {action}\nResulting observation: {obs}\n\nUpdated local progress: {local_progress}\n{'-'*50}")
        
        local_progresses.append(inside_progress)
        global_len += len(group)

    return local_progresses


def process_single_file(task_id, vari, input_dir, output_dir):
    """处理单个文件的函数，用于多线程调用"""
    try:
        input_path = os.path.join(input_dir, vari)
        output_path = os.path.join(output_dir, vari)

        with open(input_path, "r") as f:
            data = json.load(f)

        local_progress = generate_local_progress(data)
        data['local_progress'] = local_progress

        with open(output_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
            
        return f"Task {task_id} - {vari} done"
    except Exception as e:
        raise IndexError(f"Error processing {task_id}/{vari}: {str(e)}")


if __name__ == "__main__":
    MAX_WORKERS = 30

    all_tasks = []
    for task_id in range(6):
        # if task_id!=0:
        #     continue
        input_dir = f"dataset/alfworld/hrl_data_subtask/task{task_id}/"
        output_dir = f"dataset/alfworld/hrl_data_memory/task{task_id}/"
        os.makedirs(output_dir, exist_ok=True)

        # all the files need to be processed
        vari_files = [f for f in os.listdir(input_dir) if f.endswith(".json")]
        # vari_files = [f for f in os.listdir(input_dir) if f.endswith(".json") and int(f.replace(".json", "").replace("variation", "")) <= 3]
        for vari in vari_files:
            all_tasks.append((task_id, vari, input_dir, output_dir))
        
        
    print(f"Starting {len(all_tasks)} files using {MAX_WORKERS} threads...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {
            executor.submit(process_single_file, t_id, vari, inp, out): f"Task{t_id}-{vari}"
            for t_id, vari, inp, out in all_tasks
        }
        total=len(all_tasks) 
        for future in tqdm(as_completed(future_to_file), total=len(all_tasks), desc="Processing"):
            vari = future_to_file[future]
            try:
                result = future.result()
                # with print_lock:
                #     print(result)
            except Exception as exc:
                tqdm.write(f'{vari} generated an exception: {exc}')
        # result = process_single_file(task_id, "variation12.json", input_dir, output_dir)
        # print(result)
        # break
      