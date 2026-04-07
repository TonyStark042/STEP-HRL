import os
import GPUtil
import yaml
import random
import numpy as np
import torch


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

    with open("./config/step_rl.yaml", 'r') as f:
        args = yaml.safe_load(f)

    random.seed(args['seed'])
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if args['benchmark'] == 'alfworld':
        # At least 6 GPUs are needed for 6 types of ALFworld tasks evaluation, if can't meet, please modify eval function in step_bc.py
        from alg.step_rl import ALFWorld_RLstep as RL
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1" 
        os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "3600"
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    else:
        from alg.step_rl import RLstep as RL

    agent = RL(args)
    agent.update()


if __name__ == "__main__":
    main(debug=False)

