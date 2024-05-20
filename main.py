import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
from collections import deque
from env_grid import EnvOppositeV4


class Actor(nn.Module):
    def __init__(self, dim_action):
        super(Actor, self).__init__()
        self.dim_action = dim_action
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, self.dim_action)

    def get_action(self, h):
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        h = F.softmax(self.fc3(h), dim=1)
        m = Categorical(h.squeeze(0))
        a = m.sample()
        log_prob = m.log_prob(a)
        if len(a.shape) == 1:
            return a, h, log_prob
        else:
            return a.item(), h, log_prob


class Discriminator(nn.Module):
    def __init__(self, dim_state, dim_action, args):
        super(Discriminator, self).__init__()
        self._args = args
        self.dim_state = dim_state
        self.dim_action = dim_action
        _dim = self.dim_state + self.dim_action if not self._args.if_state_only else self.dim_state
        self.fc1 = nn.Linear(_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state, action):
        _input = torch.cat([state, action], 1) if not self._args.if_state_only else state
        x = torch.relu(self.fc1(_input))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


class GAIL(object):
    def __init__(self, dim_state, dim_action, args, dynamics_model=None, reward_model=None):
        self._args = args

        self.dim_state = dim_state
        self.dim_action = dim_action
        self.actor1 = Actor(self.dim_action).to(args.device)
        self.disc1 = Discriminator(self.dim_state, self.dim_action, args=args).to(args.device)
        self.d1_optimizer = torch.optim.Adam(self.disc1.parameters(), lr=1e-3)
        self.a1_optimizer = torch.optim.Adam(self.actor1.parameters(), lr=1e-3)
        self.loss_fn = torch.nn.MSELoss()
        self.adv_loss_fn = torch.nn.BCELoss()
        self.gamma = 0.9

        self.dynamics_model = dynamics_model
        self.reward_model = reward_model

    def get_action(self, obs1):
        return self.actor1.get_action(torch.tensor(obs1, device=self._args.device).float())

    def int_to_tensor(self, action):
        temp = torch.zeros(1, self.dim_action)
        temp[0, action] = 1
        return temp

    def train_D(self, s1_list, a1_list, e_s1_list, e_a1_list):
        p_s1 = torch.tensor(s1_list[0], device=self._args.device).float()
        p_a1 = self.int_to_tensor(a1_list[0]) if not self._args.if_state_only else None
        for i in range(1, len(s1_list)):
            temp_p_s1 = torch.tensor(s1_list[i], device=self._args.device).float()
            p_s1 = torch.cat([p_s1, temp_p_s1], dim=0)
            if not self._args.if_state_only:
                temp_p_a1 = self.int_to_tensor(a1_list[i])
                p_a1 = torch.cat([p_a1, temp_p_a1], dim=0)

        e_s1 = torch.tensor(e_s1_list[0], device=self._args.device).float()
        e_a1 = self.int_to_tensor(e_a1_list[0]) if not self._args.if_state_only else None
        for i in range(1, len(e_s1_list)):
            temp_e_s1 = torch.tensor(e_s1_list[i], device=self._args.device).float()
            e_s1 = torch.cat([e_s1, temp_e_s1], dim=0)
            if not self._args.if_state_only:
                temp_e_a1 = self.int_to_tensor(e_a1_list[i])
                e_a1 = torch.cat([e_a1, temp_e_a1], dim=0)

        p1_label = torch.zeros(len(s1_list), 1)
        e1_label = torch.ones(len(e_s1_list), 1)

        e1_pred = self.disc1(e_s1, e_a1)
        loss = self.adv_loss_fn(e1_pred, e1_label)
        p1_pred = self.disc1(p_s1, p_a1)
        loss = loss + self.adv_loss_fn(p1_pred, p1_label)
        self.d1_optimizer.zero_grad()
        loss.backward()
        self.d1_optimizer.step()

    def train_G(self, s1_list, a1_list, log_pi_a1_list, r1_list, e_s1_list, e_a1_list):
        # Get evaluate from Discriminator
        T = len(s1_list)
        p_s1 = torch.tensor(s1_list[0], device=self._args.device).float()
        p_a1 = self.int_to_tensor(a1_list[0]) if not self._args.if_state_only else None
        for i in range(1, len(s1_list)):
            temp_p_s1 = torch.tensor(s1_list[i], device=self._args.device).float()
            p_s1 = torch.cat([p_s1, temp_p_s1], dim=0)
            if not self._args.if_state_only:
                temp_p_a1 = self.int_to_tensor(a1_list[i])
                p_a1 = torch.cat([p_a1, temp_p_a1], dim=0)
        p1_pred = self.disc1(p_s1, p_a1)
        
        if self._args.if_as:
            s = torch.tensor(np.asarray(s1_list)[:, 0, :], device=self._args.device).float()
            a = torch.tensor(np.asarray(a1_list)[:, None], device=self._args.device).float()
            ns_pred = self.dynamics_model(torch.cat([s, a], axis=1))
            na_pred, _, _ = self.actor1.get_action(ns_pred)
            nr_pred = self.reward_model(torch.cat([ns_pred, na_pred[:, None]], axis=1))

        # Compile loss for Policy
        a1_loss = torch.FloatTensor([0.0])
        for t in range(T):
            if self._args.if_as:
                a1_loss = a1_loss + p1_pred[t, 0] * log_pi_a1_list[t] + nr_pred[t, 0]
            else:
                a1_loss = a1_loss + p1_pred[t, 0] * log_pi_a1_list[t]            
        a1_loss = -a1_loss / T

        self.a1_optimizer.zero_grad()
        a1_loss.backward()
        self.a1_optimizer.step()
    
    def save_model(self):
        torch.save(self.actor1, './output/GAIL_actor.pkl')
        torch.save(self.disc1, './output/GAIL_disc.pkl')

    def load_model(self):
        self.actor1 = torch.load('./output/GAIL_actor.pkl')
        self.disc1 = torch.load('./output/GAIL_disc.pkl')


class REINFORCE(object):
    def __init__(self, dim_action, args):
        self.dim_action = dim_action
        self._args = args
        self.actor1 = Actor(self.dim_action).to(args.device)

    def get_action(self, obs):
        return self.actor1.get_action(torch.tensor(obs, device=self._args.device).float())

    def train(self, a1_list, pi_a1_list, r_list):
        a1_optimizer = torch.optim.Adam(self.actor1.parameters(), lr=1e-3)
        T = len(r_list)
        G_list = torch.zeros(1, T, device=self._args.device)
        G_list[0, T - 1] = torch.tensor([r_list[T - 1]], device=self._args.device).float()
        for k in range(T - 2, -1, -1):
            G_list[0, k] = r_list[k] + 0.95 * G_list[0, k + 1]

        a1_loss = torch.tensor([0.0], device=self._args.device).float()
        for t in range(T):
            a1_loss = a1_loss + G_list[0, t] * torch.log(pi_a1_list[t][0, a1_list[t]])
        a1_loss = -a1_loss / T
        a1_optimizer.zero_grad()
        a1_loss.backward()
        a1_optimizer.step()

    def save_model(self):
        torch.save(self.actor1, './output/expert_actor.pkl')

    def load_model(self):
        self.actor1 = torch.load('./output/expert_actor.pkl')


if __name__ == '__main__':
    import os
    if not os.path.exists("./output"):
        os.makedirs("./output")

    torch.set_num_threads(1)
    env = EnvOppositeV4(9)

    parser = argparse.ArgumentParser(description="Argument parser for the script.")    
    parser.add_argument('--seed', type=int, default=2024, help='Random seed.')
    parser.add_argument('--wandb', action='store_true', help='Flag to use wandb.')
    parser.add_argument('--device', type=str, default='cpu', help='Run name.')
    parser.add_argument('--run_name', type=str, default='as', help='Run name.')
    parser.add_argument('--group_name', type=str, default='test', help='Group name.')
    parser.add_argument('--max_epi_iter', type=int, default=5000, help='Maximum episode iterations.')
    parser.add_argument('--max_MC_iter', type=int, default=100, help='Maximum Monte Carlo iterations.')
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of training epochs.')
    parser.add_argument('--if_as', action='store_true', help='Flag for if_as.')
    parser.add_argument('--if_no_train_expert', action='store_true', help='Flag to train expert.')
    parser.add_argument('--if_roll_expert', action='store_true', help='Flag to roll expert.')
    parser.add_argument('--if_train_agent', action='store_true', help='Flag to train agent.')
    parser.add_argument('--if_state_only', action='store_true', help='Flag for state only.')
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "cuda":
        cudnn.benchmark = True
    
    if args.wandb:
        import wandb
        wandb.login()
        wandb.init(project=args.wb_project, entity=args.wb_entity, name=args.wb_run, group=args.wb_group, dir="/tmp/wandb")
        wandb.config.update(args)

    expert = REINFORCE(dim_action=5, args=args)
    if not args.if_no_train_expert:
        # train expert policy by REINFORCE algorithm
        success_rate = deque(maxlen=100)
        for epi_iter in range(args.max_epi_iter):
            state = env.reset()
            a1_list = []
            pi_a1_list = []
            r_list = []
            acc_r = 0
            for MC_iter in range(args.max_MC_iter):
                action1, pi_a1, log_prob1 = expert.get_action(state)
                a1_list.append(action1)
                pi_a1_list.append(pi_a1)
                next_state, reward, done, info = env.step([action1, 0])
                state = next_state
                acc_r += reward
                r_list.append(reward)
                if done:
                    break                    
            success_rate.append(done)
            if args.wandb:
                wandb.log(data={
                    "expert/train_ep_return": acc_r,
                    "expert/train_reward_per_step": acc_r / MC_iter,
                    "expert/success_rate": np.mean(success_rate),
                    "custom_step": epi_iter
                    })
            if (epi_iter % 100 == 0):
                print(f"[Expert: {epi_iter}/{args.max_epi_iter}] ep_return: {acc_r: .3f}, reward_per_step: {acc_r / MC_iter: .3f}, SR: {np.mean(success_rate): .3f}")
            if done:
                expert.train(a1_list, pi_a1_list, r_list)
        adf
        expert.save_model()
    else:
        expert.load_model()

    # record expert policy
    exp_s_list = []
    exp_a_list = []
    exp_r_list = []
    expert_frames = list()
    state = env.reset()
    for MC_iter in range(args.max_MC_iter):
        action1, pi_a1, log_prob1 = expert.get_action(state)
        exp_s_list.append(state)
        exp_a_list.append(action1)
        expert_frames.append(env.render())
        next_state, reward, done, _ = env.step([action1, 0])
        exp_r_list.append(reward)
        state = next_state
        if (MC_iter % 100 == 0):
            print('step', MC_iter, 'expert 1 at', exp_s_list[MC_iter], 'expert 1 action', exp_a_list[MC_iter], 'reward', reward, 'done', done)
        if done:
            break
    
    if args.if_as:
        # Define the model
        dynamics_model = nn.Sequential(
            nn.Linear(3, 64),  # 2 state dimensions + 1 action dimension
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)   # Output is the next state dimension (2)
        )
        reward_model = nn.Sequential(
            nn.Linear(3, 64),  # 2 state dimensions + 1 action dimension
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)   # Output is the next state dimension (2)
        )

        # Define the loss function and optimizer
        loss_fn = nn.MSELoss()
        opt_d = optim.Adam(dynamics_model.parameters(), lr=0.001)
        opt_r = optim.Adam(reward_model.parameters(), lr=0.001)

        # Training loop
        for epoch in range(args.num_epochs):
            # sample a batch
            idx = np.random.randint(0, len(exp_a_list)-1, 32)
            b_state = torch.tensor(np.asarray(exp_s_list)[idx][:, 0, :], device=args.device).float()
            b_action = torch.tensor(np.asarray(exp_a_list)[idx][:, None], device=args.device).float()
            b_reward = torch.tensor(np.asarray(exp_r_list)[idx][:, None], device=args.device).float()
            b_next_state = torch.tensor(np.asarray(exp_s_list)[idx+1][:, 0, :], device=args.device).float()
            # print(b_state.shape, b_action.shape, b_next_state.shape)

            batch_d = torch.cat([b_state, b_action], dim=1)
            pred_d = dynamics_model(batch_d)
            loss_d = loss_fn(b_next_state, pred_d)
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            batch_r = torch.cat([b_state, b_action], dim=1)
            pred_r = reward_model(batch_r)
            loss_r = loss_fn(b_reward, pred_r)
            opt_r.zero_grad()
            loss_r.backward()
            opt_r.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'[{epoch + 1}/{args.num_epochs}], d-loss: {loss_d.item():.4f}, r-loss: {loss_r.item():.4f}')

        print("Training complete")

    # generative adversarial imitation learning from [exp_s_list, exp_a_list]
    if args.if_as:
        agent = GAIL(dim_state=2, dim_action=5, args=args, dynamics_model=dynamics_model, reward_model=reward_model)
    else:
        agent = GAIL(dim_state=2, dim_action=5, args=args)
    
    if args.if_train_agent:
        for epi_iter in range(args.max_epi_iter):
            state = env.reset()
            s1_list = []
            a1_list = []
            r1_list = []
            log_pi_a1_list = []
            acc_r = 0
            for MC_iter in range(args.max_MC_iter):
                action1, pi_a1, log_prob1 = agent.get_action(state)
                s1_list.append(state)
                a1_list.append(action1)
                log_pi_a1_list.append(log_prob1)
                next_state, reward, done, _ = env.step([action1, 0])
                state = next_state
                acc_r += reward
                r1_list.append(reward)
                if done:
                    break
            if args.wandb:
                wandb.log(data={"agent-train_ep_return": acc_r / MC_iter, "custom_step": epi_iter})
            if (epi_iter % 100 == 0):
                print(f"[Agent: {epi_iter}/{args.max_epi_iter}] ep_return: {acc_r: .3f}, reward_per_step: {acc_r / MC_iter: .3f}, SR: {np.mean(success_rate): .3f}")
            # train Discriminator
            agent.train_D(s1_list, a1_list, exp_s_list, exp_a_list)

            # train Generator
            agent.train_G(s1_list, a1_list, log_pi_a1_list, r1_list, exp_s_list, exp_a_list)
        agent.save_model()
    else:
        agent.load_model()

    # learnt policy
    print('expert trajectory')
    import imageio
    imageio.mimsave("./output/expert.gif", expert_frames)
    if args.wandb:
        wandb.log({"video/expert": wandb.Video("expert.gif", fps=4, format="gif")}, step=args.max_epi_iter)

    print('learnt trajectory')
    state = env.reset()
    frames = list()
    for MC_iter in range(args.max_MC_iter):
        frames.append(env.render())
        action1, pi_a1, log_prob1 = agent.get_action(state)
        exp_s_list.append(state)
        exp_a_list.append(action1)
        next_state, reward, done, _ = env.step([action1, 0])
        state = next_state
        # print('step', MC_iter, 'agent 1 at', exp_s_list[MC_iter], 'agent 1 action', exp_a_list[MC_iter])
        if done:
            break
    imageio.mimsave("./output/agent.gif", frames)
    if args.wandb:
        wandb.log({"video/agent": wandb.Video("agent.gif", fps=4, format="gif")}, step=args.max_epi_iter)
