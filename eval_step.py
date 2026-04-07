import os
import GPUtil
import yaml
import datetime
import random
import numpy as np
import torch
import torch.distributed as dist
dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=120))


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

    with open("./config/eval_step.yaml", 'r') as f:
        args = yaml.safe_load(f)

    random.seed(args['seed'])
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if args['benchmark'] == 'alfworld':
        from alg.eval_step import EvalAgent_alfworld as EvalAgent
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"  #
        os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "3600"
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    else:
        from alg.eval_step import EvalAgent

    agent = EvalAgent(args)
    agent.eval()


if __name__ == "__main__":
    main(debug=False)