import json
import os
import deepspeed
import torch
import wandb
from util.model import Policy
from util.replay_buffer import HierarchyDataset, batch_step_process
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from tqdm import tqdm 
from util.extract import extract_action_done
import pandas as pd
from scienceworld import ScienceWorldEnv
from env.scienceworld.convert_data_memory import process_obs
from util.concat_input import get_high_input, get_low_input, get_progress_input
import re
import alfworld
import alfworld.agents.environment as environment
import alfworld.agents.modules.generic as generic
import yaml


class BC:
    def __init__(self, args):
        self.args = args
        hierarcy_policy = Policy(args)
        self.engine, _ , _, _ = deepspeed.initialize(model=hierarcy_policy,
                                                     model_parameters=[{"params": [p for p in hierarcy_policy.base.parameters() if p.requires_grad], 
                                                                        "lr": args["lr"]}],
                                                     config=args["ds_config"])
        
        self.buffer = HierarchyDataset(args)
        self.eval_env = ScienceWorldEnv("", envStepLimit=args['env_step_limit'])
        self.task_names = self.eval_env.getTaskNames()

        # Initialize wandb only and save hyperparameters on rank 0
        if self.engine.global_rank == 0:
            with open(args['ds_config'], 'r') as f:
                ds_config = json.load(f)
            all_args = {**args, **ds_config}

            wandb.init(
                project=args.get('project_name', 'STEP-HRL'),
                name=f"{args['benchmark']}-{args['alg_name']}-{args['model_name']}",
                config=all_args)

        self.global_step = torch.tensor(0, dtype=torch.int64).to(self.engine.device)
        self.checkpoint_dir = f"{args['check_path']}/{args['benchmark']}/{args['alg_name']}/{args['model_name']}"

    def extract_valid_action_probs(self, log_probs, masks, max_action_nums):
        """
        Args:
            log_probs: (batch, seq_len-1)
            masks: (batch, seq_len-1), 1 is action token position
            max_action_nums: int, maximum number of actions
        Returns:
            valid_action_probs: (batch, max_action_nums)
        """
        batch_size = log_probs.size(0)
        valid_action_probs = torch.zeros(batch_size, max_action_nums, device=log_probs.device)
        
        for i in range(batch_size):
            action_positions = torch.where(masks[i]==1)[0]
            
            action_groups = []
            current_group = []
            for pos in action_positions:
                if not current_group or pos==current_group[-1]+1:
                    current_group.append(pos)
                else:
                    action_groups.append(current_group)
                    current_group = [pos]
            if current_group:
                action_groups.append(current_group)

            # average group log_prob
            for j, group in enumerate(action_groups):
                group_probs = log_probs[i, group]
                # valid_action_probs[i, j] = group_probs.sum() # Joint probability of whole action tokens
                valid_action_probs[i, j] = group_probs.sum() / len(group) # Average log_prob of action tokens

        return valid_action_probs

    def _cal_loss(self, input_tokens):
        log_probs, masks = self.engine.get_log_prob(input_tokens)
        valid_log_prob = self.extract_valid_action_probs(log_probs, masks,
                                                        max(input_tokens['action_end_mask'].sum(dim=1)))
        loss = -valid_log_prob.mean()
        self.engine.backward(loss)
        loss_item = loss.item()
        return loss_item

    def update_policy(self):
        batch_size_per_gpu = self.engine.train_micro_batch_size_per_gpu()
        sampler = DistributedSampler(self.buffer, 
                                     num_replicas=self.engine.world_size, 
                                     rank=self.engine.local_rank)
        dataloader = DataLoader(self.buffer, 
                                batch_size=batch_size_per_gpu, 
                                sampler=sampler,
                                collate_fn=HierarchyDataset.collate_fn)
        for epoch in range(self.args['epochs']):
            high_epoch_loss, low_epoch_loss, process_epoch_loss = 0.0, 0.0, 0.0
            high_loss_item, low_loss_item = 0.0, 0.0
            sampler.set_epoch(epoch)    # each epoch shuffle

            if self.engine.local_rank == 0:
                pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.args['epochs']}", leave=True)
                iterator = pbar
            else:
                iterator = dataloader
            
            for batch in iterator:
                # --- High Level (Low freq) ---
                high_interval = 4
                if self.global_step % high_interval == 0:
                    batch_high_tokens = batch_step_process(batch['high']['obs'],
                                                        batch['high']['subtask'],
                                                        self.engine.tokenizer).to(self.engine.device)
                    high_loss_item = self._cal_loss(batch_high_tokens)

                # process_data
                batch_process_tokens = batch_step_process(batch['progress']['obs'],
                                                      batch['progress']['progress'],
                                                      self.engine.tokenizer).to(self.engine.device)                
                process_loss_item = self._cal_loss(batch_process_tokens)
                
                # low Level
                batch_low_tokens = batch_step_process(batch['low']['obs'],
                                                        batch['low']['action'],
                                                        self.engine.tokenizer).to(self.engine.device)
                low_loss_item = self._cal_loss(batch_low_tokens)

                self.engine.step()

                high_epoch_loss += high_loss_item
                low_epoch_loss += low_loss_item
                process_epoch_loss += process_loss_item
                self.global_step += 1

                # Evaluation
                if self.global_step % self.args["eval_freq"] == 0 and epoch > 0:
                    eval_score = self.eval(all=self.args.get("eval_all", False))
                    if self.engine.local_rank == 0:
                        wandb.log({
                        'eval/score': eval_score,
                        'global_step': self.global_step.item()
                    })
                        self.save_policy(eval_score)

                if self.engine.local_rank == 0:
                    pbar.set_postfix({
                        'high': f"{high_loss_item:.3f}", 
                        'low': f"{low_loss_item:.3f}",
                        'process': f"{process_loss_item:.3f}",
                        'total': f"{(high_loss_item + low_loss_item + process_loss_item):.3f}"
                    })
                    wandb.log({
                        'step_loss/high': high_loss_item,
                        'step_loss/low': low_loss_item,
                        'step_loss/process': process_loss_item,
                        'step_loss/bc': (high_loss_item + low_loss_item + process_loss_item),
                        'global_step': self.global_step.item()
                    })

            if self.engine.local_rank == 0:
                print(f"hierarcy-train; epoch{epoch}, high:{high_epoch_loss}, low:{low_epoch_loss}, process:{process_epoch_loss}")
                wandb.log({
                    'epoch_loss/high': high_epoch_loss,
                    'epoch_loss/low': low_epoch_loss,
                    'epoch_loss/process': process_epoch_loss,
                    'epoch_loss/bc': high_epoch_loss + low_epoch_loss + process_epoch_loss,
                    'epoch': epoch
                })
        
        # eval_score = self.eval()
        if self.engine.local_rank == 0:
            print(f"Training done; epoch{epoch}, high:{high_epoch_loss}, low:{low_epoch_loss}, process:{process_epoch_loss}, eval_score:{eval_score}")
            # self.save_policy(eval_score)
            
    def save_policy(self, eval_score):
        save_dir = self.checkpoint_dir + "_batch"+str(self.engine.train_batch_size()) + "_step"+str(self.global_step.item()) + f"_test{eval_score:.3f}" # _score

        if self.engine.local_rank == 0:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            if self.args.get("use_lora", False):
                self.engine.module.base.save_pretrained(save_dir)
                self.engine.module.tokenizer.save_pretrained(save_dir)
                print(f"LoRA Model saved to {save_dir}")
            else:
                model_path = os.path.join(save_dir, "policy.pth")
                torch.save(self.engine.module.base.state_dict(), model_path)
                self.engine.module.tokenizer.save_pretrained(save_dir)
                print(f"Full Model saved to {model_path}")

    def run_single_eval(self, task_id, vari_id):
        episode_steps = 0
        task_name = self.task_names[task_id]
        self.eval_env.load(task_name, vari_id)

        obs, info = self.eval_env.reset()
        task_description = self.eval_env.taskdescription()        
        
        done = False
        
        # --- Memory State Initialization ---
        historical_subtasks = []
        last_subtask_progress = "None"
        
        while not done:
            # --- High-Level Input Construction ---
            current_hl_obs = process_obs(obs, info['look'], info['inv'])
            current_high_input = get_high_input(task_description, historical_subtasks, last_subtask_progress, current_hl_obs)    
            high_input_token = self.engine.tokenizer(current_high_input, return_tensors='pt').to(self.engine.device)
            with torch.no_grad():
                subtask = self.engine.generate_action(high_input_token)[0]
            
            # Update History
            historical_subtasks.append(subtask)

            # --- Low-Level Loop ---
            subtask_done = False
            current_local_progress = "None" # Initial LP for new subtask
            
            while not subtask_done:                
                curr_ll_obs = process_obs(obs, info['look'], info['inv'])
                current_input_str = get_low_input(subtask, current_local_progress, curr_ll_obs)
                low_group_token = self.engine.tokenizer(current_input_str, return_tensors='pt').to(self.engine.device)
                with torch.no_grad():
                    raw_action = self.engine.generate_action(low_group_token)[0]
                action, subtask_done = extract_action_done(raw_action)
                                
                # Execute Action
                obs_, reward, done, info = self.eval_env.step(action)
                if done:
                    break
                resulting_obs = process_obs(obs_, info['look'], info['inv'])
                
                # --- Local Progress Update ---                
                prog_input_str = get_progress_input(subtask, current_local_progress, action, resulting_obs)
                prog_token = self.engine.tokenizer(prog_input_str, return_tensors='pt').to(self.engine.device)         
                with torch.no_grad():
                    current_local_progress = self.engine.generate_action(prog_token, mode="local_progress")[0]

                # Update Obs for next step
                obs = obs_
                
                episode_steps += 1
                if episode_steps >= self.args.get('env_step_limit', 50):
                    done = True
                    break
            
            # Update Last Subtask Progress for next High-Level step
            last_subtask_progress = current_local_progress
        
        return max(0, info['score']), episode_steps

    def eval(self, dev_or_test='test', all=False):
        self.engine.eval()

        if self.engine.global_rank == 0:
            tqdm.write("Starting evaluation...")

        vari_nums = pd.read_csv(f"env/{self.args['benchmark']}/task_nums.csv", encoding='utf-8')[dev_or_test].tolist() # dev
        all_tasks = []  # [(task_id, vari_id), ...]

        for task_id, total_vari in enumerate(vari_nums):
            task_name = self.task_names[task_id]
            self.eval_env.load(task_name)
            if dev_or_test == 'dev':
                vari_ids = self.eval_env.getVariationsDev()
            elif dev_or_test == 'test':
                vari_ids = self.eval_env.getVariationsTest()
            total_vari = min(total_vari, len(vari_ids))
            if all is False:
                total_vari = total_vari // 4  # test quarter for quick eval

            selected = vari_ids[:total_vari]
            for vid in selected:
                all_tasks.append((task_id, vid))

        world_size = self.engine.world_size
        rank = self.engine.global_rank

        my_tasks = [t for i, t in enumerate(all_tasks) if i % world_size == rank]

        local_sum = 0.0
        local_count = 0
        
        # [task_id, vari_id, score, steps]
        local_details = []

        iterator = my_tasks
        if self.engine.local_rank == 0:
            iterator = tqdm(my_tasks, desc=f"Eval (Step {self.global_step.item()})", leave=False)

        for task_id, vari_id in iterator:
            try:
                score, steps = self.run_single_eval(task_id, vari_id)
            except Exception as e:
                if self.engine.local_rank == 0:
                    tqdm.write(f"[Error] Task {task_id} Vari {vari_id}: {e}")
                score = 0.0
                steps = 0
            
            local_sum += score
            local_count += 1
            local_details.append([int(task_id), int(vari_id), float(score), int(steps)])

        #  collect results from all gpus
        device = self.engine.device
        tensor = torch.tensor([local_sum, float(local_count)], device=device)
        if dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        total_sum, total_count = tensor.tolist()
        avg_score = total_sum / total_count if total_count > 0 else 0.0

        if dist.is_initialized():
            all_details_list = [None for _ in range(world_size)]
            dist.all_gather_object(all_details_list, local_details)
        else:
            all_details_list = [local_details]

        # write results
        if rank == 0:
            tqdm.write(f"[Eval] Finished at global_step {self.global_step.item()}, avg_score = {avg_score:.4f}")
            
            # [[rank0_data], [rank1_data]] -> [data, data]
            flat_details = [item for sublist in all_details_list for item in sublist] # type: ignore
            
            save_path = f"{self.checkpoint_dir}/eval_details_step{self.global_step.item()}_{dev_or_test}{avg_score:.3f}.json"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(flat_details, f, indent=4)
            
            table = wandb.Table(columns=["Task_ID", "Vari_ID", "Score", "Steps"], data=flat_details)
            wandb.log({
                "eval/detailed_table": table, 
                "eval/score": avg_score,
                "global_step": self.global_step.item()
            },)

        self.engine.train()
        return avg_score


# Differs from evaluation
class ALFWorldBC(BC):
    def __init__(self, args):
        self.args = args
        hierarcy_policy = Policy(args)
        self.engine, _ , _, _ = deepspeed.initialize(model=hierarcy_policy,
                                                     model_parameters=[{"params": [p for p in hierarcy_policy.base.parameters() if p.requires_grad], 
                                                                        "lr": args["lr"]}],
                                                     config=args["ds_config"])
        
        self.buffer = HierarchyDataset(args)

        if self.engine.global_rank == 0:
            with open(args['ds_config'], 'r') as f:
                ds_config = json.load(f)
            all_args = {**args, **ds_config}

            wandb.init(
                project=args.get('project_name', 'STEP-HRL'),
                name=f"{args['benchmark']}-{args['alg_name']}-{args['model_name']}",
                config=all_args)

        self.global_step = torch.tensor(0, dtype=torch.int64).to(self.engine.device)
        self.checkpoint_dir = f"{args['check_path']}/{args['benchmark']}/{args['alg_name']}/{args['model_name']}"
        
    def run_single_eval(self, env, game_file_path):
        """
        Runs evaluation on a single game instance.
        Args:
            env: The initialized AlfWorld environment.
            game_file_path: The specific game file to load (if env supports direct loading) 
                            OR we assume env is already reset to the correct game.
        """
        obs, info = env.reset()
        # print("Eval:", env.gamefiles.index(info['extra.gamefile'][0]))
        # return 0,0
        # match = re.search(r"(json_[^/]+/.+?)/game\.tw-pddl$", game_file_path)
        # result = match.group(1) if match else None

        # assert result in info['extra.gamefile'][0], "Game file mismatch during eval."
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
            
            while not subtask_done:                
                curr_ll_obs = process_obs(obs, mode=self.args["benchmark"])
                current_input_str = get_low_input(subtask, current_local_progress, curr_ll_obs)
                low_group_token = self.engine.tokenizer(current_input_str, return_tensors='pt').to(self.engine.device)
                with torch.no_grad():
                    raw_action = self.engine.generate_action(low_group_token)[0]
                action, subtask_done = extract_action_done(raw_action)
                                
                # Execute Action
                # AlfWorld expects a list of actions
                obs_, reward, done, info = env.step([action])
                obs_ = obs_[0] + f" {room_desc}"

                reward = reward[0]
                done = done[0]
                
                if done:
                    score = info.get('won', [False])[0] * 1.0 # AlfWorld usually returns boolean won
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
                    break
            
            last_subtask_progress = current_local_progress
        
        return score, episode_steps

    def eval(self, dev_or_test='test', all=False):
        self.engine.eval()

        if self.engine.global_rank == 0:
            tqdm.write("Starting evaluation...")

        with open("env/alfworld/base_config.yaml", "r") as f:
            eval_config = yaml.safe_load(f)

        local_sum = 0.0
        local_count = 0
        local_details = []

        rank = self.engine.global_rank
        world_size = self.engine.world_size

        if rank in [0,1,2,3,4,5]:
            # Modify this if less than 6 gpus
            # e.g. for 3 gpus, eval_config['env']['task_types'] = [rank+1, rank+1+3]
            eval_config['env']['task_types'] = [rank+1]
            eval_config['env']['expert_timeout_steps'] = self.args.get('env_step_limit', 50)
            
            if dev_or_test == 'test':
                train_eval = 'eval_out_of_distribution'
            else:
                train_eval = 'eval_in_distribution'

            EnvCls = getattr(alfworld.agents.environment, eval_config["env"]["type"])
            env_manager = EnvCls(eval_config, train_eval=train_eval)
            env = env_manager.init_env(batch_size=1)
            num_games = len(env.gamefiles)
            
            all_game_indices = list(range(num_games))
            
            if not all:
                all_game_indices = all_game_indices[:20] # Evaluate first 20 games only

            iterator = all_game_indices
            if self.engine.local_rank == 0:
                iterator = tqdm(all_game_indices, desc=f"Eval (Step {self.global_step.item()})", leave=False)
            
            for target_idx in iterator:
                try:
                    score, steps = self.run_single_eval(env, env.gamefiles[target_idx])
                    
                except Exception as e:
                    if self.engine.local_rank == 0:
                        tqdm.write(f"[Error] Game Idx {target_idx}: {e}")
                    score = 0.0
                    steps = 0
                    # If error, maybe re-init env to be saf
                
                local_sum += score
                local_count += 1
                local_details.append([rank+1, int(target_idx), float(score), int(steps)])

        # Collect Results
        device = self.engine.device
        tensor = torch.tensor([local_sum, float(local_count)], device=device)
        if dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        total_sum, total_count = tensor.tolist()
        avg_score = total_sum / total_count if total_count > 0 else 0.0

        if dist.is_initialized():
            all_details_list = [None for _ in range(world_size)]
            dist.all_gather_object(all_details_list, local_details)
        else:
            all_details_list = [local_details]

        if rank == 0:
            tqdm.write(f"[Eval] Finished at global_step {self.global_step.item()}, avg_score = {avg_score:.4f}")
            flat_details = [item for sublist in all_details_list for item in sublist] # type: ignore
            
            save_path = f"{self.checkpoint_dir}/eval_details_step{self.global_step.item()}_{dev_or_test}{avg_score:.3f}.json"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(flat_details, f, indent=4)
            
            table = wandb.Table(columns=["Task_id", "Game_Idx", "Score", "Steps"], data=flat_details)
            wandb.log({
                "eval/detailed_table": table, 
                "eval/score": avg_score,
                "global_step": self.global_step.item()
            },)

        self.engine.train()
        return avg_score