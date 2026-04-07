import alfworld.agents.environment as environment
import alfworld.agents.modules.generic as generic
import yaml
import alfworld
import os
from tqdm import tqdm
import json

with open("env/alfworld/base_config.yaml", "r") as f:
    config = yaml.safe_load(f)
with open("env/alfworld/variation_mapping.json", "r") as f:
    MAP = json.load(f)

task = config['env']['task_types'][0]


task_var = MAP[str(task)]

EnvCls = getattr(alfworld.agents.environment, config["env"]["type"])
env_manager = EnvCls(config, train_eval="train")
env = env_manager.init_env(batch_size=1)
skip_num = 0
# env.skip(skip_num)

fail_task = []

for i in tqdm(range(308)):  # 或 3553 # 790/308/650/459/533/813;
    if i < skip_num:
        continue
    obs_list = []
    next_obs_list = []
    obs, info = env.reset()
    gf = info["extra.gamefile"][0]
    gf = gf.replace("/home/tony/.cache/alfworld/", '').replace('/game.tw-pddl', '')
    if gf != "json_2.1.1/train/look_at_obj_in_light-Pillow-None-DeskLamp-316/trial_T20190908_232421_645610" and gf != "json_2.1.1/train/look_at_obj_in_light-Laptop-None-DeskLamp-323/trial_T20190908_014241_887170":
        continue
    gf = "data/" + gf
    try:
        idx = list(task_var.values()).index(gf)
        assert idx in [0,1]
        with open(f'dataset/alfworld/subtask_data/task{task-1}/variation{idx}.json', 'r') as f:
            data = json.load(f)
            actions = data['action']

        for action in actions:
            obs_list.append(obs[0])
            obs, reward, done, info = env.step([action])
            next_obs_list.append(obs[0])
            done = done[0]
            # print("Action:", action)
            # print("Observation:", obs[0])
            # print("Reward:", reward[0])
            # print("Done:", done)

        if done:
            data['obs'] = obs_list
            data['next_obs'] = next_obs_list
            data['reward'] = [0] * (len(actions) -1) + [1]
            data['done'] = [False] * (len(actions) - 1) + [True]
            os.makedirs(f'dataset/alfworld/hrl_data/task{task-1}', exist_ok=True)
            with open(f'dataset/alfworld/hrl_data/task{task-1}/variation{idx}.json', 'w') as f:
                json.dump(data, f, indent=4)
        else:
            print(f"Failed at task {task-1}, variation {idx}")
            fail_task.append({"task": task, "variation": idx, "gamefile": gf, "gamefile_id":i})
            with open('failed_tasks.jsonl', 'a') as f:
                f.write(json.dumps(fail_task[-1], ensure_ascii=False) + "\n")
    except Exception as e:
        with open('failed_tasks.jsonl', 'a') as f:
            f.write(json.dumps({"task": task, "variation": None, "gamefile": gf, "gamefile_id":i}, ensure_ascii=False) + "\n")
        print(e)

with open("all_fails.json", 'w') as f:
    json.dump(fail_task, f, indent=4)