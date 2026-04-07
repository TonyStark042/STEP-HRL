import json
import os
import re
import deepspeed
import torch
import pandas as pd
from scienceworld import ScienceWorldEnv
from util.model import Policy
from util.extract import extract_action_done
from util.concat_input import get_low_input, get_progress_input, get_high_input
from env.scienceworld.convert_data_memory import process_obs
import warnings
from transformers import AutoTokenizer
from collections import defaultdict
from alg.step_bc import BC, ALFWorldBC
os.environ["WANDB_MODE"] = "disabled"
warnings.filterwarnings("ignore", category=FutureWarning, module="peft")


class EvalAgent(BC):
    def __init__(self, args):
        self.args = args
        
        self.policy = Policy(args)
        self.engine, _, _, _ = deepspeed.initialize(
            model=self.policy,
            config=args["ds_config"]
        )
        if args['alg_name'] != "collection":
            self.checkpoint_dir = f"{args['check_path']}/{args['benchmark']}/{args['alg_name']}/{args['model_name']}"
        self.eval_env = ScienceWorldEnv("", envStepLimit=args['env_step_limit'])
        self.task_names = pd.read_csv(f"env/{self.args['benchmark']}/task_nums.csv", encoding='utf-8')['task-name'].tolist()

    def load_policy(self, path):
        if os.path.exists(path):
            if self.args["use_lora"]:
                # load LoRA weights
                self.engine.module.base.load_adapter(path, adapter_name="default", is_trainable=False)
            else:
                # Load normal model weights
                model_path = os.path.join(path, "policy.pth")
                if os.path.exists(model_path):
                    self.engine.module.base.load_state_dict(torch.load(model_path))
            
            # Load tokenizer
            self.engine.module.tokenizer = AutoTokenizer.from_pretrained(path)
            print(f"Model loaded from {path}")
        else:
            print(f"No checkpoint found at {path}")
        self.engine.eval()

    def eval(self):
        self.load_policy(self.args['load_path'])
        self.checkpoint_dir = self.args['load_path']
        self.global_step = torch.tensor(0).to(self.engine.device)
        super().eval(self.args['dev_or_test'], all=self.args.get('all', False))
    
    def data_collect(self, task_id, vari_id):
        """
        data_container:{
            'task_description',
            'subtask':[subtask_nums, steps],
            'action':[subtask_nums, steps],
            'obs':[subtask_nums, steps+1],
            'next_obs
            'reward':[subtask_nums, steps],
            'score':[subtask_nums, steps],
            'done':[subtask_nums, steps],
            'group_action':[subtask_nums, ],
            'look':[subtask_nums, steps+1],
            'inv':[subtask_nums, steps+1]
        }
        """

        episode_steps = 0
        task_name = self.task_names[task_id]
        self.eval_env.load(task_name, vari_id)

        obs, info = self.eval_env.reset()
        task_description = self.eval_env.taskdescription()
        
        local_data_container = defaultdict()

        done = False
        raw_action_list = list()
        obs_list = list()
        next_obs_list = list()
        reward_list = list()
        score_list = list()
        done_list = list()
        group_action_list = list()
        look_list = list()
        inv_list = list()
        local_progress_list = list()
        
        look_list.append(info['look'])
        inv_list.append(info['inv'])
        # --- Memory State Initialization ---
        historical_subtasks = []
        last_subtask_progress = "None"
        
        while not done:
            # --- High-Level Input Construction ---
            # Format: Historical subtasks: [...]\nLast subtask progress: ...\nCurrent observation: ...
            current_hl_obs = process_obs(obs, info['look'], info['inv'])
            current_high_input = get_high_input(task_description, historical_subtasks, last_subtask_progress, current_hl_obs)
            
            high_input_token = self.engine.tokenizer(current_high_input, return_tensors='pt').to(self.engine.device)
        
            with torch.no_grad():
                subtask = self.engine.generate_action(high_input_token)[0]
            
            # Update History
            historical_subtasks.append(subtask)

            # --- Low-Level Loop ---
            subtask_done = False
            current_local_progress = "None"
            group_action = list()
            current_local_progr_list = list()
            
            while not subtask_done: 
                obs_list.append(obs)
                current_local_progr_list.append(current_local_progress)

                curr_ll_obs = process_obs(obs, info['look'], info['inv'])
                current_input_str = get_low_input(subtask, current_local_progress, curr_ll_obs)

                low_group_token = self.engine.tokenizer(current_input_str, return_tensors='pt').to(self.engine.device)
                with torch.no_grad():
                    raw_action = self.engine.generate_action(low_group_token)[0]

                action, subtask_done = extract_action_done(raw_action)
                # subtask_done = if_done(subtask, current_local_progress, action)
                raw_action_list.append(action)
                group_action.append(action)
                
                # Execute Action
                obs_, reward, done, info = self.eval_env.step(action)
                next_obs_list.append(obs_)
                reward_list.append(reward/100)
                score_list.append(info['score']/100)
                done_list.append(done)
                look_list.append(info['look'])
                inv_list.append(info['inv'])

                if done:
                    group_action_list.append(group_action)
                    local_progress_list.append(current_local_progr_list) # Ending, don't need next local progress
                    break
                
                # --- Local Progress Update ---            
                resulting_obs = process_obs(obs_, info['look'], info['inv'])
                prog_input_str = get_progress_input(subtask, current_local_progress, action, resulting_obs)
                prog_token = self.engine.tokenizer(prog_input_str, return_tensors='pt').to(self.engine.device)
                
                with torch.no_grad():
                    # Generate next local progress
                    current_local_progress = self.engine.generate_action(prog_token, mode="local_progress")[0]

                # Update Obs for next step
                obs = obs_
                
                episode_steps += 1
                if episode_steps >= self.args.get('env_step_limit', 50):
                    done = True
                    group_action_list.append(group_action)
                    local_progress_list.append(current_local_progr_list)
                    break
                if subtask_done:
                    group_action_list.append(group_action)
                    current_local_progr_list.append(current_local_progress) # subtask done but overall not done, still need to record last local progress for next subtask
                    local_progress_list.append(current_local_progr_list)
                
            
            # Update Last Subtask Progress for next High-Level step
            last_subtask_progress = current_local_progress

        local_data_container['meta'] = (task_id, vari_id, info['score'], episode_steps)
        local_data_container['task_description'] = task_description
        local_data_container['subtask'] = historical_subtasks
        local_data_container['action'] = raw_action_list
        local_data_container['obs'] = obs_list
        local_data_container['next_obs'] = next_obs_list
        local_data_container['reward'] = reward_list
        local_data_container['score'] = score_list
        local_data_container['done'] = done_list
        local_data_container['group_action'] = group_action_list
        local_data_container['look'] = look_list
        local_data_container['inventory'] = inv_list
        local_data_container['local_progress'] = local_progress_list

        return local_data_container
    

class EvalAgent_alfworld(EvalAgent, ALFWorldBC):
    def __init__(self, args):
        self.args = args
        self.policy = Policy(args)
        self.engine, _, _, _ = deepspeed.initialize(
            model=self.policy,
            config=args["ds_config"]
        )
        if args['alg_name'] != "collection":
            self.checkpoint_dir = f"{args['check_path']}/{args['benchmark']}/{args['alg_name']}/{args['model_name']}"
    
    def data_collect(self, env, game_file_path):
        """
        data_container:{
            'task_description',
            'subtask':[subtask_nums, steps],
            'action':[subtask_nums, steps],
            'obs':[subtask_nums, steps+1],
            'next_obs
            'reward':[subtask_nums, steps],
            'score':[subtask_nums, steps],
            'done':[subtask_nums, steps],
            'group_action':[subtask_nums, ],
            'look':[subtask_nums, steps+1],
            'inv':[subtask_nums, steps+1]
        }
        """

        obs, info = env.reset()
        raw_welcome = obs[0]

        task_match = re.search(r"Your task is to: (.*)", raw_welcome)
        if task_match:
            task_description = task_match.group(1).strip()
        else:
            task_description = raw_welcome
            print(f"[Error] No 'Task_description is to' found in {game_file_path}")

        look_match = re.search(r"(Looking quickly around you,.*?)(?=\n\n)", raw_welcome, re.DOTALL)
        assert look_match, f"[Error] No 'Looking quickly around' found in {game_file_path}"
        room_desc = look_match.group(1).strip()
        obs = room_desc
        
        done = False
        episode_steps = 0
        score = 0.0
        
        local_data_container = defaultdict()

        done = False
        raw_action_list = list()
        obs_list = list()
        next_obs_list = list()
        reward_list = list()
        done_list = list()
        group_action_list = list()
        local_progress_list = list()
        
        # --- Memory State Initialization ---
        historical_subtasks = []
        last_subtask_progress = "None"
        
        while not done:
            # --- High-Level Input Construction ---
            current_hl_obs = process_obs(obs, mode=self.args["benchmark"])
            current_high_input = get_high_input(task_description, historical_subtasks, last_subtask_progress, current_hl_obs)    
            high_input_token = self.engine.tokenizer(current_high_input, return_tensors='pt').to(self.engine.device)
            with torch.no_grad():
                subtask = self.engine.generate_action(high_input_token)[0]
            
            historical_subtasks.append(subtask)

            # --- Low-Level Loop ---
            subtask_done = False
            current_local_progress = "None"
            group_action = list()
            current_local_progr_list = list()
            
            while not subtask_done:
                obs_list.append(obs)
                current_local_progr_list.append(current_local_progress)

                curr_ll_obs = process_obs(obs, mode=self.args["benchmark"])
                current_input_str = get_low_input(subtask, current_local_progress, curr_ll_obs)
                low_group_token = self.engine.tokenizer(current_input_str, return_tensors='pt').to(self.engine.device)
                with torch.no_grad():
                    raw_action = self.engine.generate_action(low_group_token)[0]

                action, subtask_done = extract_action_done(raw_action)
                raw_action_list.append(action)
                group_action.append(action)
                                
                # Execute Action
                # AlfWorld expects a list of actions
                obs_, reward, done, info = env.step([action])
                obs_ = obs_[0] + f" {room_desc}"
                reward = reward[0]
                done = done[0]

                next_obs_list.append(obs_)
                reward_list.append(reward)
                done_list.append(done)
                
                if done:
                    score = info.get('won', [False])[0] * 1.0 # AlfWorld usually returns boolean won
                    group_action_list.append(group_action)
                    local_progress_list.append(current_local_progr_list) # Ending, don't need next local progress
                    break

                resulting_obs = process_obs(obs_, mode=self.args["benchmark"])
                
                # --- Local Progress Update ---                
                prog_input_str = get_progress_input(subtask, current_local_progress, action, resulting_obs)
                prog_token = self.engine.tokenizer(prog_input_str, return_tensors='pt').to(self.engine.device)         
                with torch.no_grad():
                    current_local_progress = self.engine.generate_action(prog_token, mode="local_progress")[0]

                obs = obs_
                episode_steps += 1

                if episode_steps >= self.args.get('env_step_limit', 50):
                    done = True
                    group_action_list.append(group_action)
                    local_progress_list.append(current_local_progr_list) # Ending, don't need next local progress
                    break
                if subtask_done:
                    group_action_list.append(group_action)
                    current_local_progr_list.append(current_local_progress)
                    local_progress_list.append(current_local_progr_list)
            
            last_subtask_progress = current_local_progress

        local_data_container['meta'] = (game_file_path, score, episode_steps)
        local_data_container['task_description'] = task_description
        local_data_container['subtask'] = historical_subtasks
        local_data_container['action'] = raw_action_list
        local_data_container['obs'] = obs_list
        local_data_container['next_obs'] = next_obs_list
        local_data_container['reward'] = reward_list
        local_data_container['done'] = done_list
        local_data_container['group_action'] = group_action_list
        local_data_container['local_progress'] = local_progress_list

        return local_data_container