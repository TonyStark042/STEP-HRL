import alfworld.agents.environment as environment
import alfworld.agents.modules.generic as generic
import yaml
import alfworld
import os
from tqdm import tqdm
import json
import re

taskId = 1

# load fail expert
fails = []
with open("failed_tasks.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            fails.append(item)

# load env
with open("env/alfworld/base_config.yaml", "r") as f:
    config = yaml.safe_load(f)
config['env']['task_types'] = [taskId]
EnvCls = getattr(alfworld.agents.environment, config["env"]["type"])
env_manager = EnvCls(config, train_eval="train")

def _ini_env(skip):    
    env = env_manager.init_env(batch_size=1)
    env.skip(skip)
    return env

# load failed expert
def _load_fdata(task_id, var_id, gamefile):
    with open(f'dataset/alfworld/subtask_data/task{task_id-1}/variation{var_id}.json', 'r') as f:
        data = json.load(f)
        # actions = data['action']
    assert data['game_file'] in gamefile, f"Game mismatch at {task_id}, {var_id}"
    return data

def _return_empty_var():
    pattern = re.compile(r'variation(\d+)\.json')
    existing_ids = set()

    for filename in os.listdir(f'dataset/alfworld/hrl_data/task{taskId-1}'):
        match = pattern.match(filename)
        if match:
            existing_ids.add(int(match.group(1)))
    current_id = 0
    while True:
        if current_id not in existing_ids:
            return current_id
        current_id += 1

clean_fails = [i for i in fails if i['task'] == taskId]

for i in tqdm(clean_fails):
    # if i['task'] == taskId and i['variation'] != None and f"variation{i['variation']}.json" not in exist_var
    # data = _load_fdata(i['task'], i['variation'], i['gamefile'])
    # f_actions = data['action']
    # if i['variation'] == None:
    if i['task'] != taskId:
        continue
    i['ori_variation'] = i['variation']
    i['variation'] = _return_empty_var()
    skip = i['gamefile_id']
    env = _ini_env(skip=skip)
    data = {"id": f"alfworld_{i['variation']}",
            "game_file": i['gamefile'],
            "gamefile_id": skip,
            "task_description": '',
            "subtask": [],
            "action": [],
            "obs": [],
            "next_obs": [],
            "reward": [],
            "done": [],
            "group_action": [],
            "local_progress": []}

    obs_list = []
    next_obs_list = []
    real_actions = []
    obs, info = env.reset()
    assert i['gamefile'].replace("data/", '') in info['extra.gamefile'][0], "Gamefile mismatch"
    steps = 0
    idx = 0
    done = False
    while not done and steps < 30:
        a = info['extra.expert_plan'][0][0]
        # a = f_actions[idx]
        if a != "look":
            obs_list.append(obs[0])
        obs, reward, done, info = env.step([a])
        # if obs[0] == "Nothing happens.":
        #     a = info['extra.expert_plan'][0][0]
        #     obs, reward, done, info = env.step([a])
        # else:
        #     idx +=1
        if a != "look":
            steps += 1
            next_obs_list.append(obs[0])
            real_actions.append(a)

        done = done[0]
    i['steps'] = steps
    if done:
        print(f"Fixed task{i['task']}, var{i['variation']}, steps: {steps}")
        data['action'] = real_actions
        data['obs'] = obs_list
        data['next_obs'] = next_obs_list
        data['reward'] = [0] * (len(obs_list) -1) + [1]
        data['done'] = [False] * (len(obs_list) - 1) + [True]
        with open(f'dataset/alfworld/hrl_data/task{i["task"]-1}/variation{i["variation"]}.json', 'w') as f:
            json.dump(data, f, indent=4)
        with open('fixed_tasks.jsonl', 'a') as f:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")
    else:
        with open('fail2_tasks.jsonl', 'a') as f:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")
        print(f"Still Failed at task{i['task']}, var{i['variation']}")


