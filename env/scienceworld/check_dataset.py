from scienceworld import ScienceWorldEnv
import json
import os
from argparse import ArgumentParser
import pandas as pd

TASK_NUM = 30
env = ScienceWorldEnv("", envStepLimit=1000)
task_names = env.getTaskNames()
df = pd.read_csv("env/scienceworld/task_nums.csv")

# parser = ArgumentParser()
# parser.add_argument('--task_id', '-t', type=int, default=10)
# args = parser.parse_args()
# task_id = args.task_id

for task_id in df["task-id"]:
    # if task_id<12:
    #     continue
    env.load(task_names[task_id], generateGoldPath=False)
    # vari_ids = env.getVariationsTrain()
    for vari in range(int(df[df['task-id'] == task_id]["train"])):
        env.load(task_names[task_id], vari)
        with open(f'dataset/scienceworld/hrl_data_memory/task{task_id}/variation{vari}.json', 'r') as json_file:
            raw_data = json.load(json_file)

        action_traj = []
        for group in raw_data['group_action']:
            assert len(group) > 0, f"{task_id}, {vari}, empty group action"
            for a in group:
                action_traj.append(a)

        assert action_traj == raw_data['action'], f"{task_id}, {vari}, action traj != action"
        assert len(raw_data['action']) == len(action_traj), f"{task_id}, {vari}, group_action len:{len(action_traj)}, action len:{len(raw_data['action'])}"
        assert len(raw_data['action']) == len(raw_data['obs']) 
        assert len(raw_data['subtask']) == len(raw_data['group_action'])
        assert len(raw_data['local_progress']) == len(raw_data['group_action'])
        assert len(raw_data['inventory']) == len(raw_data['look']) # look & inv is one more than others
        assert len(raw_data['action']) + 1 == len(raw_data['look']), f"{task_id}, {vari}, action len+1:{len(raw_data['action'])+1}, look len:{len(raw_data['look'])}"
        assert len(raw_data['obs']) == len(raw_data['next_obs'])
        assert len(raw_data['obs']) == len(raw_data['done'])
        assert len(raw_data['obs']) == len(raw_data['reward'])
        assert len(raw_data['reward']) == len(raw_data['score'])
        assert True in raw_data['done'] , f"{task_id}, {vari}, never done"
        assert raw_data['score'][-1] >= 1.0 , f"{task_id}, {vari}, done but last score {raw_data['score'][-1]}"

        all_group_action = list()
        for i in range(len(raw_data['group_action'])):
            ga = raw_data['group_action'][i]
            lp = raw_data['local_progress'][i]
            if i != len(raw_data['group_action']) -1: 
                assert len(ga)+1 == len(lp), f"{task_id}, {vari}, step {i}, group_action len:{len(ga)}, local_progress len:{len(lp)}"
            else:
                assert len(ga) == len(lp), f"{task_id}, {vari}, step {i}, group_action len:{len(ga)}, local_progress len:{len(lp)}"
            all_group_action.extend(ga)
        assert all_group_action == action_traj, f"{task_id}, {vari}, all group action != action traj"

        print(task_id, vari, "check done")
    else:
        print(f"task {task_id} check all variations done!")