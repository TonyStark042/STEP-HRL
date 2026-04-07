from scienceworld import ScienceWorldEnv
import json
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


TASK_NUM = 30

def main(range_start, range_end):
    env = ScienceWorldEnv("", envStepLimit=1000)
    df = pd.read_csv("env/scienceworld/task_nums.csv")
    task_names = env.getTaskNames()
    data_type = "train"

    for task_id in range(range_start, range_end):
        # if task_id not in [0]:
        #         continue
        env.load(task_names[task_id], generateGoldPath=True)
        if data_type == "train":
            # vari_ids = env.getVariationsTrain()
            vari_ids = range(df["train"][task_id])
        
        path = f"dataset/scienceworld/hrl_data_memory/task{task_id}/"
        os.makedirs(path, exist_ok=True)
        for vari in vari_ids:
            if vari not in [12]:
                continue
            env.load(task_names[task_id], vari, generateGoldPath=True)
            task_description = env.taskdescription()
            with open(f'dataset/scienceworld/hrl_data_memory/task{task_id}/variation{vari}.json', 'r') as json_file:
                gold_data = json.load(json_file)
            gold_data["obs"] = []
            gold_data["look"] = []
            gold_data["inventory"] = []
            gold_data["next_obs"] = []
            gold_data["reward"] = []
            gold_data["score"] = []
            gold_data["done"] = []
            
            if isinstance(gold_data["action"][0],str):
                action_traj = gold_data["action"]
                # gold_data["group_action"] = []
            else:
                action_traj = []
                for traj in gold_data['action']:
                    for a in traj:
                        action_traj.append(a)
                gold_data["group_action"] = gold_data["action"]
                gold_data["action"] = action_traj

            obs, infos = env.reset()
            for action in action_traj:
                gold_data["obs"].append(obs)
                gold_data["look"].append(infos["look"])
                gold_data["inventory"].append(infos["inv"])

                obs_, reward, isCompleted, infos = env.step(action)
                
                gold_data["reward"].append(reward/100)
                gold_data["score"].append(infos["score"]/100)
                gold_data["done"].append(isCompleted)
                gold_data["next_obs"].append(obs_)
                obs = obs_
            else:
                gold_data["look"].append(infos["look"])
                gold_data["inventory"].append(infos["inv"])

            # print(gold_data)
            
            with open(path+f'variation{vari}.json', 'w') as json_file:
                json.dump(gold_data, json_file, indent=4)
                print(task_id, vari,"   done")
            # exit()

if __name__ == "__main__":
    # ranges = [
    #     (0,5),
    #     (5,6),
    #     (6,7),
    #     (7,8),
    #     (8,9),
    #     (9,17),
    #     (17,21),
    #     (21,27),
    #     (27,28),
    #     (28,29),
    #     (29,30)
    # ]

    # futures = []
    # with ProcessPoolExecutor(max_workers=32) as exe:
    #     for r in ranges:
    #         rs = r[0]
    #         re = r[1]
    #         futures.append(exe.submit(main, rs, re))

    #     for f in as_completed(futures):
    #         try:
    #             f.result()
    #         except Exception as e:
    #             print(f"[ERROR] worker failed: {e}")
    main(9, 10)
