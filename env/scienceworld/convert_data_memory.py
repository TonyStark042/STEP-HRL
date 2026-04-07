import pandas as pd
from util.concat_input import get_done_input, get_high_input, get_low_input, get_progress_input
from argparse import ArgumentParser
import json
import os

def process_obs(raw_obs:str, look:str=None, inventory:str=None, mode="scienceworld") -> str:
    if mode == "scienceworld":
        raw_obs = str(raw_obs) if raw_obs is not None else ""
        look = str(look) if look is not None else "None"
        inventory = str(inventory) if inventory is not None else "None"
        
        out_str = "\nObs: " + raw_obs + "\n"        
        out_str += "Look: " + look + "\n"
        out_str += "Inventory: " + inventory
    else:
        out_str = raw_obs + "\n"
    return out_str 

def convert_data(half: int, benchmark: str, specific_file_path=None):
    raw_data_buffer = {
        "task_description":[], "subtask":[], "obs": [], "action": [], "group_action":[],
        "next_obs": [], "look": [], "inventory": [], "reward": [], "score": [], "done": [],
        "local_progress": []
    }
    
    # 1. Low-Level Dataset
    low_dataset = {
        "obs":[], # Context (Local Progress + Subtask, Obs) 
        "next_obs": [],            
        "action":[],  # Target Action
        "reward":[], "score":[], "done":[]
    }
    
    # 2. High-Level Dataset
    high_dataset = {
        "obs":[],              # Context (History + Last Progress + Obs)
        "subtask":[],          # Subtask, action,
        "next_obs": [],        
        # 'next_subtask': [],    # RL only fot high-level
        "reward":[], "score":[], "done":[]
    }

    # 3. Local Progress Dataset
    progress_dataset = {
        "obs": [],    # Context (Subtask + Prev LP + Action + Result Obs)
        "next_obs": [],            
        "reward": [],
        "done": [],
        "progress": []  # Target (Next LP)
    }

    if specific_file_path is None:
        vari_nums = pd.read_csv(f"env/{benchmark}/task_nums.csv",encoding='utf-8')['train'].tolist()
        
        for task_id, vari_num in enumerate(vari_nums):
            if vari_num == 0: continue
                
            if half == 0:
                start_idx, step = 0, 2
            elif half == 1:
                start_idx, step = 1, 2
            else: 
                start_idx, step = 0, 1
            for vari_id in range(start_idx, vari_num, step):
                # if task_id != 0 or vari_id != 0: continue
                path = f"dataset/{benchmark}/hrl_data_memory/task{task_id}/variation{vari_id}.json"
                if not os.path.exists(path): 
                    path = f"dataset/{benchmark}/hrl_data/task{task_id}/variation{vari_id}.json"
                    if not os.path.exists(path): continue
                    
                with open(path, 'r') as f:
                    raw_traj = json.load(f)

                for key in raw_data_buffer.keys():
                    if key in raw_traj:
                        raw_data_buffer[key].append(raw_traj[key])
                    else:
                        raw_data_buffer[key].append([])
    else:
        with open(specific_file_path, 'r') as f:
            raw_traj = json.load(f)
        # raw_traj = raw_traj[:1]
        for item in raw_traj:
            for key in item.keys():
                if key in raw_data_buffer:
                    raw_data_buffer[key].append(item[key])

    print(f"Processing {len(raw_data_buffer['task_description'])} trajectories...")

    # --- processing ---
    for i in range(len(raw_data_buffer['task_description'])):
        global_step = 0
        if specific_file_path is not None and sum(map(len, raw_data_buffer['group_action'][i])) > 50:
            raw_data_buffer['group_action'][i].pop()
            raw_data_buffer['local_progress'][i].pop()
            assert sum(map(len, raw_data_buffer['group_action'][i])) == 50
        
        my_high_obs = []
        my_low_obs = []
        my_lp_obs = []

        # iterate each Subtask (Group)
        for group_id, group_action in enumerate(raw_data_buffer['group_action'][i]):
            
            # ==========================================
            # Part A: High-Level Dataset Construction
            # ==========================================
            
            # 1. Historical Subtasks
            hist_subtasks = raw_data_buffer['subtask'][i][:group_id]
            
            # 2. Last Subtask Progress
            if group_id == 0:
                last_subtask_progress = "None"
            else:
                # Get the last element of the local_progress of the previous subtask
                prev_lp_list = raw_data_buffer['local_progress'][i][group_id-1]
                last_subtask_progress = prev_lp_list[-1] if len(prev_lp_list) > 0 else "None"
            
            # 3. Current Observation
            current_hl_obs = process_obs(raw_data_buffer['obs'][i][global_step], 
                                        raw_data_buffer['look'][i][global_step], 
                                        raw_data_buffer['inventory'][i][global_step])
            
            high_obs = get_high_input(raw_data_buffer['task_description'][i], hist_subtasks, last_subtask_progress, current_hl_obs)
            high_dataset['obs'].append(high_obs)
            my_high_obs.append(high_obs)
            high_dataset['subtask'].append(raw_data_buffer['subtask'][i][group_id])
            
            #  High-Level Reward
            group_reward = sum(raw_data_buffer['reward'][i][global_step : global_step + len(group_action)])
            group_score = sum(raw_data_buffer['score'][i][global_step : global_step + len(group_action)])
            # Group Action Done
            group_done = raw_data_buffer['done'][i][global_step + len(group_action) - 1]
            
            high_dataset['reward'].append(group_reward)
            high_dataset['score'].append(group_score)
            high_dataset['done'].append(group_done)

            # ==========================================
            # Part B: Low-Level & Progress Dataset
            # ==========================================
            
            current_subtask_str = raw_data_buffer['subtask'][i][group_id]
            current_lp_list = raw_data_buffer['local_progress'][i][group_id]
            
            # Iterate each action within the Group
            for k, action_str in enumerate(group_action):
                
                # --- 1. Low-Level Data ---
                # Get the current Local Progress corresponding to the action (index k)
                curr_lp = current_lp_list[k] if k < len(current_lp_list) else "None"
                
                curr_ll_obs = process_obs(raw_data_buffer['obs'][i][global_step], 
                                          raw_data_buffer['look'][i][global_step], 
                                          raw_data_buffer['inventory'][i][global_step])
                
                is_last_step = (k == len(group_action) - 1)
                action_with_status = f"{action_str}; {is_last_step}"
                
                low_obs = get_low_input(current_subtask_str, curr_lp, curr_ll_obs)
                low_dataset['obs'].append(low_obs)
                my_low_obs.append(low_obs)
                low_dataset['action'].append(action_with_status)
                step_reward = 1.0 if is_last_step else 0.0
                # check illegal action
                if raw_data_buffer['next_obs'][i][global_step] == "No known action matches that input.":
                    step_reward = -1.0
                low_dataset['reward'].append(step_reward)
                low_dataset['score'].append(step_reward)
                low_dataset['done'].append(True if is_last_step else False)
         
                # --- 2. Local Progress Dataset ---
                # Input: LP_t, Action_t, Obs_t+1
                # Target: LP_t+1
                if k + 1 < len(current_lp_list):
                    prev_lp = curr_lp # LP_t
                    target_lp = current_lp_list[k+1] # LP_t+1
                    
                    # Get Resulting Observation (Obs_t+1)
                    # if global_step + 1 < len(raw_data_buffer['obs'][i]):
                    resulting_obs = process_obs(raw_data_buffer['next_obs'][i][global_step],
                                                raw_data_buffer['look'][i][global_step + 1],
                                                raw_data_buffer['inventory'][i][global_step + 1])

                    prog_input = get_progress_input(current_subtask_str, prev_lp, action_str, resulting_obs)
                    progress_dataset['obs'].append(prog_input)
                    progress_dataset['reward'].append(step_reward)
                    progress_dataset['done'].append(is_last_step)
                    my_lp_obs.append(prog_input)
                    target_lp = target_lp.removesuffix(" || [Checked: ]").removesuffix(" ||").strip()
                    progress_dataset['progress'].append(target_lp)
            

                global_step += 1
        else:
            if 0< raw_data_buffer['score'][i][-1] < 1:
                high_dataset['next_obs'].extend(my_high_obs[1:] + ["Task not fully completed."])
                low_dataset['next_obs'].extend(my_low_obs[1:] + [get_low_input(raw_data_buffer['subtask'][i][-1], "Task failed.", process_obs(raw_data_buffer['next_obs'][i][-1], raw_data_buffer['look'][i][-1], raw_data_buffer['inventory'][i][-1]))])
                progress_dataset['next_obs'].extend(my_lp_obs[1:] + ["Task failed."])

            elif raw_data_buffer['score'][i][-1] == 1:
                high_dataset['next_obs'].extend(my_high_obs[1:] + ["Task successfully finished"])
                low_dataset['next_obs'].extend(my_low_obs[1:] + [get_low_input(raw_data_buffer['subtask'][i][-1], "Task successfully finished", process_obs(raw_data_buffer['next_obs'][i][-1], raw_data_buffer['look'][i][-1], raw_data_buffer['inventory'][i][-1]))])
                progress_dataset['next_obs'].extend(my_lp_obs[1:] + ["Task successfully finished"])
            else:
                high_dataset['next_obs'].extend(my_high_obs[1:] + ["Task totalsly failed."])
                low_dataset['next_obs'].extend(my_low_obs[1:] + [get_low_input(raw_data_buffer['subtask'][i][-1], "Task failed.", process_obs(raw_data_buffer['next_obs'][i][-1], raw_data_buffer['look'][i][-1], raw_data_buffer['inventory'][i][-1]))])
                progress_dataset['next_obs'].extend(my_lp_obs[1:] + ["Task failed."])
            
            if raw_data_buffer['score'][i][-1] != 1 and len(raw_data_buffer['obs'][i]) == 50: # env_limit_step
                progress_dataset['reward'][-1] = 0.0
                progress_dataset['done'][-1] = False
                
            progress_dataset['done'][-1] = low_dataset['done'][-1]
            progress_dataset['reward'][-1] = low_dataset['reward'][-1]

            # high_dataset['next_subtask'].extend(raw_data_buffer['subtask'][i][1:] + ["Task finished"])
                            
    # --- Save datasets ---
    os.makedirs(f"dataset/{benchmark}/low_data/", exist_ok=True)
    os.makedirs(f"dataset/{benchmark}/high_data/", exist_ok=True)
    os.makedirs(f"dataset/{benchmark}/progress_data/", exist_ok=True)
    os.makedirs(f"dataset/{benchmark}/done_data/", exist_ok=True)

    if specific_file_path is not None:
        file_name = os.path.basename(specific_file_path)
        # file_name = 'test.json'
        with open(f"dataset/{benchmark}/high_data/{file_name}", 'w') as f:
            json.dump(high_dataset, f, indent=4)
        with open(f"dataset/{benchmark}/low_data/{file_name}", 'w') as f:
            json.dump(low_dataset, f, indent=4)
        with open(f"dataset/{benchmark}/progress_data/{file_name}", 'w') as f:
            json.dump(progress_dataset, f, indent=4)
    
    suffix = f"_half{half}" if half in [0, 1] else "_expert"
    filename_low = f"low_data{suffix}.json" if half in [0, 1] else "expert.json"
    filename_high = f"high_data{suffix}.json" if half in [0, 1] else "expert.json"
    filename_prog = f"progress_data{suffix}.json" if half in [0, 1] else "expert.json"
    filename_done = f"done_data{suffix}.json" if half in [0, 1] else "expert.json"

    with open(f"dataset/{benchmark}/low_data/{filename_low}", 'w') as f:
        json.dump(low_dataset, f, indent=4)
    
    with open(f"dataset/{benchmark}/high_data/{filename_high}", 'w') as f:
        json.dump(high_dataset, f, indent=4)
        
    with open(f"dataset/{benchmark}/progress_data/{filename_prog}", 'w') as f:
        json.dump(progress_dataset, f, indent=4)

    print(f"Saved datasets to dataset/{benchmark}/...")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--benchmark', '-b', type=str, default='scienceworld', help='scienceworld or alfworld')
    parser.add_argument('--half', type=int, default=0, help='0: first half, 1: second half, -1: all data')
    parser.add_argument('--specific_file_path', '-f', type=str, default=None, help='Path to a specific file to process') # 'dataset/scienceworld/collect_data/medium1_seed1_num3840_1_correctExpert.json'
    args = parser.parse_args()
    
    convert_data(args.half, args.benchmark, args.specific_file_path)