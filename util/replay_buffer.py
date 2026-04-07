from torch.utils.data import Dataset
# import os
import json
import pandas as pd
import torch
from transformers.tokenization_utils_base import BatchEncoding
from util.inst import high_prompt, low_prompt
from collections import deque
import random

def batch_step_process(batch_states, batch_actions, tokenizer):
    batch_input, batch_mask, batch_labels = [], [], []
    batch_state_end_mask, batch_action_end_mask = [], []
    for state, action in zip(batch_states, batch_actions):
        state_token = tokenizer(state, return_tensors='pt', add_special_tokens=True)
        action_token = tokenizer(action, return_tensors='pt', add_special_tokens=False)
        action_token['input_ids'] = torch.cat([action_token['input_ids'], torch.tensor([[tokenizer.eos_token_id]])], dim=-1)
        action_token['attention_mask'] = torch.cat([action_token['attention_mask'], torch.tensor([[1]])], dim=-1)

        state_label = torch.full_like(state_token['input_ids'], -100)
        action_label = action_token['input_ids'] # label

        # record state and action end token position
        state_end_mask = torch.zeros_like(state_token['input_ids'])
        action_end_mask = torch.zeros_like(action_token['input_ids'])
        state_end_mask[0, -1], action_end_mask[0, -1] = 1, 1 # state and action end token mark as 1

        batch_input.append(torch.cat([state_token['input_ids'], action_token['input_ids']], dim=1))
        batch_mask.append(torch.cat([state_token['attention_mask'], action_token['attention_mask']], dim=1))
        batch_labels.append(torch.cat([state_label, action_label], dim=1))

        batch_state_end_mask.append(torch.cat([state_end_mask, torch.zeros_like(action_token['input_ids'])], dim=1))
        batch_action_end_mask.append(torch.cat([torch.zeros_like(state_token['input_ids']), action_end_mask], dim=1))

    # Padding, automatically pad to the longest sequence length in the batch, and pad to the right
    padded_input = torch.nn.utils.rnn.pad_sequence([x.squeeze(0) for x in batch_input], 
                                                batch_first=True, 
                                                padding_value=tokenizer.pad_token_id)
    padded_mask = torch.nn.utils.rnn.pad_sequence([x.squeeze(0) for x in batch_mask], 
                                                batch_first=True, 
                                                padding_value=0)
    padded_labels = torch.nn.utils.rnn.pad_sequence([x.squeeze(0) for x in batch_labels], 
                                                batch_first=True, 
                                                padding_value=-100)
    
    padded_state_end_mask = torch.nn.utils.rnn.pad_sequence([x.squeeze(0) for x in batch_state_end_mask],
                                                                batch_first=True,
                                                                padding_value=0)
    padded_action_end_mask = torch.nn.utils.rnn.pad_sequence([x.squeeze(0) for x in batch_action_end_mask], 
                                                                batch_first=True, 
                                                                padding_value=0)
    return BatchEncoding({
        "input_ids": padded_input,
        "attention_mask": padded_mask,
        "labels": padded_labels, # labels for calculating loss, only calculate on action tokens
        "state_end_mask": padded_state_end_mask,
        "action_end_mask": padded_action_end_mask
    })

def batch_traj_process(batch_prompt, batch_states, batch_actions, tokenizer):   # test done
        batch_input, batch_mask, batch_labels = [], [], []
        batch_state_end_mask, batch_action_end_mask = [], []
        for prompt, states, actions in zip(batch_prompt, batch_states, batch_actions):
            prompt_token = tokenizer(prompt, return_tensors='pt')
            input_tensors = [prompt_token['input_ids']]
            mask_tensors = [prompt_token['attention_mask']]
            labels = [torch.full_like(input_tensors[0], -100)]

            state_end_masks = [torch.zeros_like(input_tensors[0])]
            action_end_masks = [torch.zeros_like(input_tensors[0])]

            for state, action in zip(states[:-1], actions):
                action = action + tokenizer.eos_token
                state_token = tokenizer(state, return_tensors='pt')
                action_token = tokenizer(action, return_tensors='pt')

                state_label = torch.full_like(state_token['input_ids'], -100)
                action_label = action_token['input_ids']

                # record state and action end token position
                state_end_mask = torch.zeros_like(state_token['input_ids'])
                action_end_mask = torch.zeros_like(action_token['input_ids'])
                state_end_mask[0, -1], action_end_mask[0, -1] = 1, 1 # state and action end token mark as 1

                input_tensors.extend([state_token['input_ids'], action_token['input_ids']])
                mask_tensors.extend([state_token['attention_mask'], action_token['attention_mask']])
                labels.extend([state_label, action_label])
                state_end_masks.extend([state_end_mask, torch.zeros_like(action_token['input_ids'])])
                action_end_masks.extend([torch.zeros_like(state_token['input_ids']), action_end_mask])
            
            final_state_token = tokenizer(states[-1], return_tensors='pt')
            final_state_label = torch.full_like(final_state_token['input_ids'], -100)
            final_state_end_mask = torch.zeros_like(final_state_token['input_ids'])
            final_state_end_mask[0, -1] = 1  # Marks the end position of the last state
            
            input_tensors.append(final_state_token['input_ids'])
            mask_tensors.append(final_state_token['attention_mask'])
            labels.append(final_state_label)
            state_end_masks.append(final_state_end_mask)
            action_end_masks.append(torch.zeros_like(final_state_token['input_ids']))

            batch_input.append(torch.cat(input_tensors, dim=1))
            batch_mask.append(torch.cat(mask_tensors, dim=1))
            batch_labels.append(torch.cat(labels, dim=1))

            batch_state_end_mask.append(torch.cat(state_end_masks, dim=1))
            batch_action_end_mask.append(torch.cat(action_end_masks, dim=1))

        # Padding
        padded_input = torch.nn.utils.rnn.pad_sequence([x.squeeze(0) for x in batch_input], 
                                                    batch_first=True, 
                                                    padding_value=tokenizer.pad_token_id)
        padded_mask = torch.nn.utils.rnn.pad_sequence([x.squeeze(0) for x in batch_mask], 
                                                    batch_first=True, 
                                                    padding_value=0)
        padded_labels = torch.nn.utils.rnn.pad_sequence([x.squeeze(0) for x in batch_labels], 
                                                    batch_first=True, 
                                                    padding_value=-100)
        
        padded_state_end_mask = torch.nn.utils.rnn.pad_sequence([x.squeeze(0) for x in batch_state_end_mask],
                                                                 batch_first=True,
                                                                   padding_value=0)
        padded_action_end_mask = torch.nn.utils.rnn.pad_sequence([x.squeeze(0) for x in batch_action_end_mask], 
                                                                 batch_first=True, 
                                                                 padding_value=0)

        return BatchEncoding({
            "input_ids": padded_input,
            "attention_mask": padded_mask,
            "labels": padded_labels,
            "state_end_mask": padded_state_end_mask,
            "action_end_mask": padded_action_end_mask
        }) # prompt1, s11, a11, s12, a12, ..., prompt2, s21, a21 ...


class SequenceDataset(Dataset):
    def __init__(self, args):
        super(SequenceDataset, self).__init__()
        self.args = args
        self.data = {
            "task_description":[],
            "obs": [],
            "action": [],
            "next_obs": [],
            "reward": [],
            "score": [],
            "done": []
        }
        self.load_data()
    
    def load_data(self):
        vari_nums = pd.read_csv(f"env/{self.args['benchmark']}/task_nums.csv",encoding='utf-8')['train'].tolist()
        for task_id, vari_num in enumerate(vari_nums):
            for vari_id in range(vari_num):
                path = f"dataset/{self.args['benchmark']}/task{task_id}/variation{vari_id}.json"
                with open(path, 'r') as f:
                    raw_traj = json.load(f)
                for key in self.data.keys():
                    self.data[key].append(raw_traj[key])
        

    def __len__(self):
        return len(self.data["obs"])
    
    def __getitem__(self, idx):
        return {key: value[idx] for key, value in self.data.items()}
    
    @staticmethod
    def collate_fn(batch):
        # Custom collate_fn to handle sequences of different lengths
        batch_data = {
            'task_description': [],
            'obs': [],
            'action': [],
            'next_obs': [],
            'reward': [],
            'score': [],
            'done': []
        }
        
        for sample in batch:
            for key in batch_data:
                batch_data[key].append(sample[key])
                
        return batch_data
    

class HierarchyDataset(Dataset):
    def __init__(self, args):
        super(HierarchyDataset, self).__init__()
        self.args = args
        self.load_data()

    def load_data(self):
        if self.args.get("taskId", None) != None:
            data_dir = f"dataset/{self.args['benchmark']}/Train{self.args['taskId']}"
        else:
            data_dir = f"dataset/{self.args['benchmark']}"

        if self.args['mode'] == 'rl':
            with open(f"{data_dir}/high_data/expert.json", 'r') as f:
                self.high_data = json.load(f)
        elif self.args['mode'] == 'bc':
            if self.args['half'] not in [0,1]:
                with open(f"{data_dir}/high_data/expert.json", 'r') as f:
                    self.high_data = json.load(f)
            else:
                with open(f"{data_dir}/high_data/high_data_half{self.args['half']}.json", 'r') as f:
                    self.high_data = json.load(f)

        with open(f"{data_dir}/progress_data/expert.json", 'r') as f:
            self.progress_data = json.load(f)
        with open(f"{data_dir}/low_data/expert.json", 'r') as f: # for rl
            self.low_data = json.load(f)
            
        if self.args['mode'] == 'rl':
            for path in self.args['medium_dataset']:
                with open(f"{data_dir}/high_data/{path}", 'r') as f:
                    medium = json.load(f)
                for key in self.high_data:
                    self.high_data[key] += medium[key]

                with open(f"{data_dir}/low_data/{path}", 'r') as f:
                    medium = json.load(f)
                for key in self.low_data:
                    self.low_data[key] += medium[key]

                with open(f"{data_dir}/progress_data/{path}", 'r') as f:
                    medium = json.load(f)
                for key in self.progress_data:
                    self.progress_data[key] += medium[key]

        if hasattr(self, 'progress_data'):
            self.progress_len = len(self.progress_data['obs'])
        else:
            self.progress_len = 0

        self.low_len = len(self.low_data['obs'])
        self.high_len = len(self.high_data['obs'])
        self.data_size = max(self.low_len, self.high_len, self.progress_len)
        print(f"High: {self.high_len}, Low: {self.low_len}, Progress: {self.progress_len}")

    def __len__(self):
        return self.data_size
    
    def __getitem__(self, idx):
        high_idx = idx % self.high_len
        low_idx = idx % self.low_len
        item = {
            'high': {key: value[high_idx] for key, value in self.high_data.items()},
            'low': {key: value[low_idx] for key, value in self.low_data.items()}
        }
        
        # progress data
        if hasattr(self, 'progress_data') and self.progress_len > 0:
            progress_idx = idx % self.progress_len
            item['progress'] = {key: value[progress_idx] for key, value in self.progress_data.items()}
            
        return item
    
    @staticmethod
    def collate_fn(batch):
        batch_data = {
            'high': {key: [] for key in batch[0]['high'].keys()},
            'low': {key: [] for key in batch[0]['low'].keys()}
        }
        if 'progress' in batch[0]:
            batch_data['progress'] = {key: [] for key in batch[0]['progress'].keys()}
        if 'done' in batch[0]:
            batch_data['done'] =  {key: [] for key in batch[0]['done'].keys()}
        if 'medium' in batch[0]:
            batch_data['medium'] = {key: [] for key in batch[0]['medium'].keys()}

        for sample in batch:
            for key in batch_data['high']:
                batch_data['high'][key].append(sample['high'][key])
            for key in batch_data['low']:
                batch_data['low'][key].append(sample['low'][key])
            if 'progress' in sample:
                for key in batch_data['progress']:
                    batch_data['progress'][key].append(sample['progress'][key])
            if 'done' in sample:
                for key in batch_data['done']:
                    batch_data['done'][key].append(sample['done'][key])
            if 'medium' in sample:
                for key in batch_data['medium']:
                    batch_data['medium'][key].append(sample['medium'][key])

        return batch_data