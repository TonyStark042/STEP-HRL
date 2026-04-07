import json
from util.model import Policy, Critic
from util.replay_buffer import batch_traj_process, batch_step_process, HierarchyDataset
import wandb
from torch.utils.data import DataLoader, DistributedSampler
import deepspeed
import torch
import copy
from scienceworld import ScienceWorldEnv
import os
from tqdm import tqdm
from alg.step_bc import BC, ALFWorldBC


class ActorCritic(Policy):
    def __init__(self, args):
        super().__init__(args)
        hidden_dim = self.base.config.hidden_size
        self.high_critic = Critic(hidden_dim).to(dtype=torch.bfloat16)
        self.target_high_critic = copy.deepcopy(self.high_critic)
        self.low_critic = Critic(hidden_dim).to(dtype=torch.bfloat16)
        self.target_low_critic = copy.deepcopy(self.low_critic)
        self.lp_critic = Critic(hidden_dim).to(dtype=torch.bfloat16)
        self.target_lp_critic = copy.deepcopy(self.lp_critic)

        for model in [self.target_high_critic, self.target_low_critic, self.target_lp_critic]:
            for param in model.parameters():
                param.requires_grad = False
        self.soft_update_target_critic(tau=1.0)

    def soft_update_target_critic(self, tau):
        assert 0.0 <= tau <= 1.0
        for target_param, param in zip(self.target_high_critic.parameters(), 
                                       self.high_critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) 
                                    + param.data * tau)
        for target_param, param in zip(self.target_low_critic.parameters(), 
                                       self.low_critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) 
                                    + param.data * tau)
        for target_param, param in zip(self.target_lp_critic.parameters(), 
                                       self.lp_critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) 
                                    + param.data * tau)
    
    def critic_forward(self, x, mode):
        if mode=="high":
            v, q = self.high_critic(x)
        elif mode=="low":
            v, q = self.low_critic(x)
        elif mode=="progress":
            v, q = self.lp_critic(x)
        return v.squeeze(-1), q.squeeze(-1)
    
    @torch.no_grad()
    def target_critic_forward(self, x, mode):
        if mode == "high":
            v, q = self.target_high_critic(x)
        elif mode == "low":
            v, q = self.target_low_critic(x)
        elif mode == "progress":
            v, q = self.target_lp_critic(x)
        return v.squeeze(-1), q.squeeze(-1)
            
class RLstep(BC):
    def __init__(self, args):
        self.args = args
        actor_critic = ActorCritic(args)
        if args['bc_model'] != None:
            actor_critic.base.load_adapter(args["bc_model"], adapter_name="default", is_trainable=True) # overwrite the original adapter
            print("Loaded BC model from:", args["bc_model"])
        else:
            print("Training from scratch", args['model_name'])
        self.engine, _ , _, _ = deepspeed.initialize(
                                           model=actor_critic,
                                           model_parameters=[{"params": [p for p in actor_critic.base.parameters() if p.requires_grad], 
                                                              "lr": args["actor_lr"]},
                                                              {"params": [p for p in actor_critic.high_critic.parameters() if p.requires_grad], 
                                                               "lr": args["critic_lr"]},
                                                               {"params": [p for p in actor_critic.low_critic.parameters() if p.requires_grad], 
                                                               "lr": args["critic_lr"]},
                                                               {"params": [p for p in actor_critic.lp_critic.parameters() if p.requires_grad], 
                                                               "lr": args["critic_lr"]}],
                                           config=args["ds_config"])
        
        self.loss_fct = torch.nn.MSELoss()
        self.buffer = HierarchyDataset(args)
        self.eval_env = ScienceWorldEnv("", envStepLimit=args['env_step_limit'])
        self.task_names = self.eval_env.getTaskNames()

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
    
    def save_critic(self, eval_score):
        save_dir = self.checkpoint_dir + "_batch"+str(self.engine.train_batch_size()) + "_step"+str(self.global_step.item()) + f"_test{eval_score:.3f}" # _score
        
        if self.engine.local_rank == 0:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, "high_critic.pth")
            torch.save(self.engine.module.high_critic.state_dict(), model_path)
            model_path = os.path.join(save_dir, "low_critic.pth")
            torch.save(self.engine.module.low_critic.state_dict(), model_path)
            model_path = os.path.join(save_dir, "lp_critic.pth")
            torch.save(self.engine.module.lp_critic.state_dict(), model_path)

    def update_ac(self, batch_data, mode="high"):
        if mode == "high":
            key = "subtask"
        elif mode == 'low':
            key = "action"
        elif mode == "progress":
            key = "progress"

        # obs
        batch_tokens = batch_step_process(batch_data['obs'],
                                          batch_data[key],
                                          self.engine.tokenizer).to(self.engine.device)
        # next_obs
        next_batch_tokens = batch_step_process(batch_data['next_obs'],
                                                batch_data[key], # placeholder
                                                self.engine.tokenizer).to(self.engine.device)

        rewards, dones = self.prepare_tensor(batch_data['reward'], batch_data['done'])

        # Critic
        with torch.no_grad():
            next_hidden_states, next_state_end_mask, next_action_end_mask = self.engine.get_hidden_states(next_batch_tokens) # batch_size, sqe_len
            next_state_end_hidden = next_hidden_states[next_state_end_mask == 1] # batch_size, 1
            target_vs_next, _ = self.engine.target_critic_forward(next_state_end_hidden, mode=mode) 
            # TD target
            target = rewards + (1 - dones) * self.args['gama'] * target_vs_next 

        with torch.no_grad():
            hidden_states, state_end_mask, action_end_mask = self.engine.get_hidden_states(batch_tokens)
            state_end_hidden = hidden_states[state_end_mask == 1]
            action_end_hidden = hidden_states[action_end_mask == 1]
            _, target_qsa = self.engine.target_critic_forward(state_end_hidden, mode=mode)
            
        vs, _ = self.engine.critic_forward(state_end_hidden, mode=mode)
        _, q_sa = self.engine.critic_forward(action_end_hidden, mode=mode)

        # Critic Loss
        v_loss = vs - target_qsa
        # Conservative estimate, small
        weight = torch.where(v_loss<0,
                            torch.ones_like(vs)*(1-self.args['weight_tau']),
                            torch.ones_like(vs)*self.args['weight_tau'])
        v_loss = (weight * (v_loss**2)).mean()
        q_loss = self.loss_fct(q_sa, target)
        self.engine.backward(q_loss + v_loss)
        # self.engine.step()
        
        # Actor
        actor_loss = 0.0
        if self.global_step.item() < self.args['warmup_steps']:
            pass
            # actor_loss = self._cal_loss(batch_tokens)
        else:
            action_log_probs, action_masks = self.engine.get_log_prob(batch_tokens)
            valid_action_log_probs = self.extract_valid_action_probs(action_log_probs, action_masks, max(batch_tokens['action_end_mask'].sum(dim=1)))
            if self.args['use_adv']:
                # Advantage: Q(s,a)-V(s)
                with torch.no_grad():
                    adv = q_sa - vs
                    beta = self.args.get("awr_beta", 1.0)
                    w = torch.exp(adv / beta)
                    w = torch.clamp(w, max=self.args.get("awr_w_clip", 20.0))
                    w = w / (w.mean() + 1e-8)
                    
                actor_loss = -torch.mean(adv * valid_action_log_probs)
            else:
                actor_loss = -torch.mean(q_sa.detach()*valid_action_log_probs) # target_qsa
            self.engine.backward(actor_loss)
            actor_loss = actor_loss.item()
            # self.engine.step()
        return v_loss.item(), q_loss.item(), actor_loss
            
    def update(self,):
        batch_size_per_gpu = self.engine.train_micro_batch_size_per_gpu()
        sampler = DistributedSampler(self.buffer, 
                                     num_replicas=self.engine.world_size, 
                                     rank=self.engine.local_rank)
        
        dataloader = DataLoader(self.buffer, 
                                batch_size=batch_size_per_gpu, 
                                sampler=sampler,
                                collate_fn=HierarchyDataset.collate_fn)
        
        for epoch in range(self.args['epochs']):
            high_epoch_Vloss, high_epoch_Qloss, high_epoch_actorloss, low_epoch_Vloss, low_epoch_Qloss, low_epoch_actorloss, lp_epoch_Vloss, lp_epoch_Qloss, lp_epoch_actorloss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            sampler.set_epoch(epoch)    # each epoch shuffle

            if self.engine.local_rank == 0:
                pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.args['epochs']}", leave=True)
                iterator = pbar
            else:
                iterator = dataloader

            for batch in iterator:
                """
                Args:
                    batch:{
                        "task_description": [traj_nums,], or "subtask":[group_nums, ]
                        "obs":[traj_nums, steps+1],       or "obs":[group_nums, steps+1]
                        "subtask":[traj_nums, steps],     or "action":[group_nums, steps]
                        "reward": [traj_nums, steps]      or "reward":[group_nums, steps]
                        "done": [traj_nums, steps]        or "done": [grop_nums, steps]
                    }
                """
                # high level
                high_v_loss, high_q_loss, high_actor_loss = self.update_ac(batch['high'], mode="high")               
            
                # low level
                low_v_loss, low_q_loss, low_actor_loss = self.update_ac(batch['low'], mode="low")  

                # local progress 
                lp_v_loss, lp_q_loss, lp_actor_loss = self.update_ac(batch['progress'], mode="progress")  

                self.engine.soft_update_target_critic(tau=self.args['tau'])
                self.engine.step()
               
                high_epoch_Vloss += high_v_loss
                high_epoch_Qloss += high_q_loss
                high_epoch_actorloss += high_actor_loss
                low_epoch_Vloss += low_v_loss
                low_epoch_Qloss += low_q_loss
                low_epoch_actorloss += low_actor_loss
                lp_epoch_Vloss += lp_v_loss
                lp_epoch_Qloss += lp_q_loss
                lp_epoch_actorloss += lp_actor_loss

                self.global_step += 1
                
                if self.engine.local_rank == 0:  # Only log on the main process
                    wandb.log({
                        'step_loss/high_v': high_v_loss,
                        'step_loss/high_q': high_q_loss,
                        'step_loss/high_actor': high_actor_loss,
                        'step_loss/low_v': low_v_loss,
                        'step_loss/low_q': low_q_loss,
                        'step_loss/low_actor': low_actor_loss,
                        'step_loss/progress_v': lp_v_loss,
                        'step_loss/progress_q': lp_q_loss,
                        'step_loss/progress_actor': lp_actor_loss,
                        'global_step': self.global_step.item()
                    })
                    pbar.set_postfix({
                        'high': f"{high_actor_loss:.3f}",
                        'low': f"{low_actor_loss:.3f}",
                        'process': f"{lp_actor_loss:.3f}",
                        'total': f"{(high_actor_loss + low_actor_loss + lp_actor_loss):.3f}"
                    })

                # eval_freq = 100 if epoch < 2 else self.args['eval_freq']
                eval_freq = self.args['eval_freq']
                if self.global_step.item() % eval_freq == 0:
                    eval_score = self.eval(all=True)
                    if self.engine.local_rank == 0:
                        wandb.log({
                        'eval/score': eval_score,
                        'global_step': self.global_step.item()
                    })
                        self.save_policy(eval_score)
                        self.save_critic(eval_score)

            if self.engine.local_rank == 0:
                # print(f"hierarcy-train; epoch{epoch}, highQ:{high_epoch_Qloss}, highActor:{high_epoch_actorloss}, mediumQ:{medium_epoch_Qloss}, mediumActor:{medium_epoch_actorloss}, low:{low_epoch_loss}, process:{process_epoch_loss}")
                wandb.log({
                    'epoch_loss/high_epoch_Vloss': high_epoch_Vloss,
                    'epoch_loss/high_epoch_Qloss': high_epoch_Qloss,
                    'epoch_loss/high_epoch_actorloss': high_epoch_actorloss,
                    'epoch_loss/low_epoch_Vloss': low_epoch_Vloss,
                    'epoch_loss/low_epoch_Qloss': low_epoch_Qloss,
                    'epoch_loss/low_epoch_actorloss': low_epoch_actorloss,
                    'epoch_loss/progress_epoch_Vloss': lp_epoch_Vloss,
                    'epoch_loss/progress_epoch_Qloss': lp_epoch_Qloss,
                    'epoch_loss/progress_epoch_actorloss': lp_epoch_actorloss,
                    'epoch': epoch
                })

    def prepare_tensor(self, rewards, dones):
        reward_tensor = torch.tensor(rewards, dtype=torch.bfloat16, device=self.engine.device)
        done_tensor = torch.tensor(dones, dtype=torch.bfloat16, device=self.engine.device)
        
        return reward_tensor, done_tensor
    

class ALFWorld_RLstep(RLstep, ALFWorldBC):
    def __init__(self, args):
        self.args = args
        actor_critic = ActorCritic(args)
        if args['bc_model'] != None:
            actor_critic.base.load_adapter(args["bc_model"], adapter_name="default", is_trainable=True) # overwrite the original adapter
            print("Loaded BC model from:", args["bc_model"])
        else:
            print("Training from scratch", args['model_name'])
        self.engine, _ , _, _ = deepspeed.initialize(
                                           model=actor_critic,
                                           model_parameters=[{"params": [p for p in actor_critic.base.parameters() if p.requires_grad], 
                                                              "lr": args["actor_lr"]},
                                                              {"params": [p for p in actor_critic.high_critic.parameters() if p.requires_grad], 
                                                               "lr": args["critic_lr"]},
                                                               {"params": [p for p in actor_critic.low_critic.parameters() if p.requires_grad], 
                                                               "lr": args["critic_lr"]},
                                                               {"params": [p for p in actor_critic.lp_critic.parameters() if p.requires_grad], 
                                                               "lr": args["critic_lr"]}],
                                           config=args["ds_config"])
        
        self.loss_fct = torch.nn.MSELoss()
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