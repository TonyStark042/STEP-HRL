import os
import GPUtil
import json
import pandas as pd
import datetime
import random
import glob
import numpy as np
import torch
from tqdm import tqdm
import torch.distributed as dist
import yaml

dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=120)) # To avoid timeout in data collection

def sample(args, eval_agent, my_rank):
    my_save_path = f"dataset/{args['benchmark']}/collect_data/{my_rank}/medium_seed{args['seed']}_num{args['collect_vari_num_per_gpu']}.jsonl"
    os.makedirs(os.path.dirname(my_save_path), exist_ok=True)
    
    if args['benchmark'] == 'scienceworld':
        # Total collected samples = valid_tasks(24) * collect_vari_num_per_gpu * world_size
        vari_nums = pd.read_csv(f"env/{args['benchmark']}/task_nums.csv",encoding='utf-8')['train'].tolist()
        valid_tasks = [t for t in vari_nums if t > 0]
        total_iterations = len(valid_tasks) * args['collect_vari_num_per_gpu']

        with tqdm(total=total_iterations, desc="Collecting steps", unit="it", disable=(my_rank != 0)) as pbar:
            for task_id, vari_num in enumerate(vari_nums):
                if vari_num == 0:
                    continue

                for i in range(args['collect_vari_num_per_gpu']):
                    vari_id = random.choice(range(vari_num))
                    one_data = eval_agent.data_collect(task_id, vari_id)

                    with open(my_save_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(one_data, ensure_ascii=False) + "\n")
                        
                    pbar.update(1)
    else:
        # Total collected samples = collect_vari_num_per_gpu * 6 (At least 6 GPUs needed, if not meet, do the same as in step_bc.py)
        import alfworld
        import alfworld.agents.environment as environment
        import alfworld.agents.modules.generic as generic

        with open("env/alfworld/base_config.yaml", "r") as f:
            eval_config = yaml.safe_load(f)
        if my_rank in [0,1,2,3,4,5]:
            eval_config['env']['task_types'] = [my_rank+1]
            eval_config['env']['expert_timeout_steps'] = args.get('env_step_limit', 50)
            
            EnvCls = getattr(alfworld.agents.environment, eval_config["env"]["type"])
            env_manager = EnvCls(eval_config, train_eval="train")
            env = env_manager.init_env(batch_size=1)
            num_games = len(env.gamefiles)
            
            all_game_indices = list(range(num_games))
            all_game_indices = all_game_indices[:args["collect_vari_num_per_gpu"]]

            iterator = all_game_indices
            if my_rank == 0:
                iterator = tqdm(all_game_indices, desc=f"Collecting steps", leave=False)
            for target_idx in iterator:
                one_data = eval_agent.data_collect(env, env.gamefiles[target_idx])

                with open(my_save_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(one_data, ensure_ascii=False) + "\n")            


def main(debug=False):
    num_gpus_needed = int(os.environ.get('WORLD_SIZE', 1))
    free_gpus = GPUtil.getAvailable(order='memory', limit=num_gpus_needed, maxLoad=0.1, maxMemory=0.1, includeNan=False, excludeID=[], excludeUUID=[])
    print(f"Using GPUs: {free_gpus}")
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, free_gpus))

    if debug:
        os.environ["WANDB_MODE"] = "disabled" 
        rank = int(os.environ.get("RANK", 0))
        import debugpy

        debugpy.listen(address = ('0.0.0.0', 5678 + rank))
        if rank == 0:
            debugpy.wait_for_client() 
        breakpoint()

    with open("./config/step_collection.yaml", 'r') as f:
        args = yaml.safe_load(f)
    my_rank = dist.get_rank()
    random.seed(args['seed'] + my_rank) 
    np.random.seed(args['seed'] + my_rank)
    torch.manual_seed(args['seed'] + my_rank)

    if args['benchmark'] == 'alfworld':
        from alg.eval_step import EvalAgent_alfworld as EvalAgent
    else:
        from alg.eval_step import EvalAgent

    eval_agent = EvalAgent(args)
    assert args['collect_model_path'] is not None, "Please provide a trained model path for data collection."
    eval_agent.load_policy(args['collect_model_path'])

    sample(args, eval_agent, my_rank)
    dist.barrier()

    if my_rank == 0:
        print("Merging jsonl files...")
        final_container = list()
        
        part_files = glob.glob(f"dataset/{args['benchmark']}/collect_data/*/medium_seed{args['seed']}_num{args['collect_vari_num_per_gpu']}.jsonl")

        for p_file in part_files:
            with open(p_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    
                    single_step_data = json.loads(line)
                    final_container.append(single_step_data)
            
            os.remove(p_file)

        final_save_path = f"dataset/{args['benchmark']}/collect_data/medium_seed{args['seed']}_num{len(final_container)}.json"
        os.makedirs(os.path.dirname(final_save_path), exist_ok=True)
        with open(final_save_path, 'w', encoding='utf-8') as f:
            json.dump(final_container, f, indent=4)
        print(f"Data collection completed. Total collected samples: {len(final_container)}. Saved to {final_save_path}.")


if __name__ == "__main__":
    main(debug=False)