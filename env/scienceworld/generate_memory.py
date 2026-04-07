from openai import OpenAI
from util.inst import local_progress_prompt_final
import json, os
from scienceworld import ScienceWorldEnv
from concurrent.futures import ThreadPoolExecutor, as_completed
from env.scienceworld.convert_data_memory import process_obs
# from env.scienceworld.convert_data import process_obs
import threading
print_lock = threading.Lock()

client = OpenAI(
    api_key="--",
    base_url="https://api.deepseek.com",
)

local_progress_prompt = """You are an AI agent responsible for updating local progress within a subtask.

Based on:
- the current subtask,
- the previous local progress,
- the current action,
- the resulting observation,

update the local progress.

Local progress represents a cumulative state summary of the subtask.

State Inheritance Rule:
- Any fact that was previously established by observation and remains true MUST be retained in the updated local progress.
- Do NOT remove previously established facts unless the new observation explicitly contradicts them.
- Temporary actions (e.g., opening, looking, checking) should NOT overwrite persistent facts (e.g., objects obtained, items held, target found).

---

### MODE SELECTION
First, determine which mode applies based on the current subtask.

---

### MODE A: Search / Find Task
(Subtasks such as "find X and focus it", "locate Y". Navigation-only subtasks do NOT belong here.)

Output format:
<one short sentence about the local progress> || [Checked: loc1, loc2, ...]

Rules:
1. **Inheritance**:
   - You MUST retain all previously recorded Checked locations.
2. **Checked Update Rule**:
   - Add the current location to Checked IF:
     a) the subtask is a search/find task,
     b) a search-related action was performed (e.g., go to a new location, open a container, etc),
     c) when you go to a new location which is not in the Checked list, you MUST add it to the Checked list,
     d) the observation explicitly confirms the target (or its equivalent) is NOT present.
   - Do NOT add a location if the target is found.
3. **Target Equivalence**:
   - Treat EGGS and LARVAE as the ANIMAL itself.
   - If an egg or larva is observed, the target is considered FOUND.
4. **Found Case**:
   - If the target (or equivalent) is found, update the status accordingly.
   - You may still retain the Checked list.
4. If the target is found:
   - Clearly convey that the search is completed.
   - The sentence should sound resolved or finished.
5. If the target is NOT found:
   - The sentence MUST reflect that the search is still ongoing and the key changes in Obs, and can imply to search the unchecked locations but don't output the unchecked locations explicitly. 
   - If you want to imply, must contain "unchecked" word (like "need to search unchecked places", "ready to search an unchecked rooms"), but must NOT contain the unchecked location names.
   - If there is door opened to a new location, you MUST mention it in the updated local progress.
   Important restriction:
    - When the search is not complete, you MUST NOT mention any specific locations,
    room names, or concrete places as next targets.
    - Do NOT list, suggest, or hint at particular rooms (e.g., kitchen, bathroom, outside).
    - Use only abstract references such as "other areas", "unchecked places",
    or "elsewhere" to convey continuation.
6. Language style:
   - Avoid fixed or repetitive sentence patterns between previous local progress and the updated one.
   - Vary sentence structure across similar situations.
   - Prefer describing the agent's situation and search state
     rather than stating generic negatives like "nothing is found".
   - You can NOT start with "Searching", like "Searching for an animal ..."
---

### MODE B: Navigation Task
(Subtasks such as "go to kitchen", "navigate to garden", AFTER the agent has reached the specified search location, if any.)

Output format:
<one short sentence about the local progress> || [Route: locA -> locB -> ...]

Rules:
1. **Inheritance**:
   - Retain the existing Route if present.
2. **Route Update**:
   - Append the new location ONLY when the agent actually enters a new location.
3. **No Search Assumptions**:
   - Do NOT infer whether the target exists or not in any location.

---

### MODE C: Other Tasks
(Manipulation, usage, cooking, waiting, monitoring, etc.)

Output format:
<one short sentence about local progress>

Rules:
- Do NOT include Checked or Route.
- Focus on the current state change supported by the observation.
- Retain relevant cumulative state if applicable (e.g., heating, holding object).

---

### General Constraints
Do NOT:
- introduce information beyond what is supported by the observation,
- reason about future subtasks or overall planning,
- repeat full action or observation history,
- infer or suggest future actions.

Notes:
- The eggs and larvae of any animal are considered the animal itself.

Output the updated local progress as concise plain text following the appropriate mode format.
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

        # initial local progress
    #     obs = process_obs(data['obs'][global_len], data['look'][global_len], data['inventory'][global_len])
    #     local_progress = llm(f"""Subtask: {subtask}
    # Current observation:\n {obs}\n
    # Generated local progress:""", 
    #                          init=True)
        local_progress = "None"
        inside_progress = [local_progress]
        
        for idx2, action in enumerate(group):
            global_idx = global_len + idx2
            
            if global_idx + 1 >= len(data['obs']):
                break
            obs = process_obs(data['obs'][global_idx + 1], data['look'][global_idx + 1], data['inventory'][global_idx + 1])
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
    TASK_NUM = 30
    TASK_START = 0
    TASK_END = TASK_START + 2

    MAX_WORKERS = 20

    for task_id in range(27, 29):
        input_dir = f"dataset/scienceworld/hrl_data_memory_fixed/task{task_id}/"
        output_dir = f"dataset/scienceworld/hrl_data_memory_2/task{task_id}/"
        os.makedirs(output_dir, exist_ok=True)

        # all the files need to be processed
        # vari_files = [f for f in os.listdir(input_dir) if f.endswith(".json") and int(f.replace(".json", "").replace("variation", "")) >= 4]
        vari_files = [f for f in os.listdir(input_dir) if f.endswith(".json")]
        
        print(f"Starting Task {task_id} with {len(vari_files)} files using {MAX_WORKERS} threads...")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_file = {
                executor.submit(process_single_file, task_id, vari, input_dir, output_dir): vari 
                for vari in vari_files
            }

            for future in as_completed(future_to_file):
                vari = future_to_file[future]
                try:
                    result = future.result()
                    with print_lock:
                        print(result)
                except Exception as exc:
                    with print_lock:
                        print(f'{vari} generated an exception: {exc}')
        # result = process_single_file(task_id, "variation12.json", input_dir, output_dir)
        # print(result)
        # break
      