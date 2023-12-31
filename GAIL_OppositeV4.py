from torch.distributions.categorical import Categorical
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from env_OppositeV4 import EnvOppositeV4
import numpy as np
import csv
from collections import deque


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
    def __init__(self, dim_state, dim_action, args):
        self._args = args

        self.dim_state = dim_state
        self.dim_action = dim_action
        self.actor1 = Actor(self.dim_action)
        self.disc1 = Discriminator(self.dim_state, self.dim_action, args=args)
        self.d1_optimizer = torch.optim.Adam(self.disc1.parameters(), lr=1e-3)
        self.a1_optimizer = torch.optim.Adam(self.actor1.parameters(), lr=1e-3)
        self.loss_fn = torch.nn.MSELoss()
        self.adv_loss_fn = torch.nn.BCELoss()
        self.gamma = 0.9

    def get_action(self, obs1):
        return self.actor1.get_action(torch.from_numpy(obs1).float())

    def int_to_tensor(self, action):
        temp = torch.zeros(1, self.dim_action)
        temp[0, action] = 1
        return temp

    def train_D(self, s1_list, a1_list, e_s1_list, e_a1_list):
        p_s1 = torch.from_numpy(s1_list[0]).float()
        p_a1 = self.int_to_tensor(a1_list[0]) if not self._args.if_state_only else None
        for i in range(1, len(s1_list)):
            temp_p_s1 = torch.from_numpy(s1_list[i]).float()
            p_s1 = torch.cat([p_s1, temp_p_s1], dim=0)
            if not self._args.if_state_only:
                temp_p_a1 = self.int_to_tensor(a1_list[i])
                p_a1 = torch.cat([p_a1, temp_p_a1], dim=0)

        e_s1 = torch.from_numpy(e_s1_list[0]).float()
        e_a1 = self.int_to_tensor(e_a1_list[0]) if not self._args.if_state_only else None
        for i in range(1, len(e_s1_list)):
            temp_e_s1 = torch.from_numpy(e_s1_list[i]).float()
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
        p_s1 = torch.from_numpy(s1_list[0]).float()
        p_a1 = self.int_to_tensor(a1_list[0]) if not self._args.if_state_only else None
        for i in range(1, len(s1_list)):
            temp_p_s1 = torch.from_numpy(s1_list[i]).float()
            p_s1 = torch.cat([p_s1, temp_p_s1], dim=0)
            if not self._args.if_state_only:
                temp_p_a1 = self.int_to_tensor(a1_list[i])
                p_a1 = torch.cat([p_a1, temp_p_a1], dim=0)

        e_s1 = torch.from_numpy(e_s1_list[0]).float()
        e_a1 = self.int_to_tensor(e_a1_list[0]) if not self._args.if_state_only else None
        for i in range(1, len(e_s1_list)):
            temp_e_s1 = torch.from_numpy(e_s1_list[i]).float()
            e_s1 = torch.cat([e_s1, temp_e_s1], dim=0)
            if not self._args.if_state_only:
                temp_e_a1 = self.int_to_tensor(e_a1_list[i])
                e_a1 = torch.cat([e_a1, temp_e_a1], dim=0)
        p1_pred = self.disc1(p_s1, p_a1)

        # Compile loss for Policy
        a1_loss = torch.FloatTensor([0.0])
        for t in range(T):
            a1_loss = a1_loss + p1_pred[t, 0] * log_pi_a1_list[t]
        a1_loss = -a1_loss / T

        self.a1_optimizer.zero_grad()
        a1_loss.backward()
        self.a1_optimizer.step()
    
    def save_model(self):
        torch.save(self.actor1, 'GAIL_actor.pkl')
        torch.save(self.disc1, 'GAIL_disc.pkl')

    def load_model(self):
        self.actor1 = torch.load('GAIL_actor.pkl')
        self.disc1 = torch.load('GAIL_disc.pkl')


class REINFORCE(object):
    def __init__(self, dim_action, args):
        self.dim_action = dim_action
        self._args = args
        self.actor1 = Actor(self.dim_action)

    def get_action(self, obs):
        return self.actor1.get_action(torch.from_numpy(obs).float())

    def train(self, a1_list, pi_a1_list, r_list):
        a1_optimizer = torch.optim.Adam(self.actor1.parameters(), lr=1e-3)
        T = len(r_list)
        G_list = torch.zeros(1, T)
        G_list[0, T - 1] = torch.FloatTensor([r_list[T - 1]])
        for k in range(T - 2, -1, -1):
            G_list[0, k] = r_list[k] + 0.95 * G_list[0, k + 1]

        a1_loss = torch.FloatTensor([0.0])
        for t in range(T):
            a1_loss = a1_loss + G_list[0, t] * torch.log(pi_a1_list[t][0, a1_list[t]])
        a1_loss = -a1_loss / T
        a1_optimizer.zero_grad()
        a1_loss.backward()
        a1_optimizer.step()

    def save_model(self):
        torch.save(self.actor1, 'V4_actor.pkl')

    def load_model(self):
        self.actor1 = torch.load('V4_actor.pkl')


if __name__ == '__main__':
    torch.set_num_threads(1)
    env = EnvOppositeV4(9)
    
    class P(object):
        wandb = True
        run_name = "test"
        group_name = "test"
        # max_epi_iter = 10000
        max_epi_iter = 100
        max_MC_iter = 100
        if_train_expert = True  # False
        if_roll_expert = True  # False
        if_train_agent = True  # False
        if_state_only = True
    
    args = P()
    
    if args.wandb:
        import wandb
        wandb.login()
        wandb.init(project="img-gen-rl", entity="rowing0914", name=args.run_name, group=args.group_name, dir="/tmp/wandb")
        wandb.config.update(args)

    expert = REINFORCE(dim_action=5, args=args)
    if args.if_train_expert:
        # train expert policy by REINFORCE algorithm
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
                next_state, reward, done, _ = env.step([action1, 0])
                state = next_state
                acc_r = acc_r + reward
                r_list.append(reward)
                if done:
                    break
            if args.wandb:
                wandb.log(data={"expert/train_ep_return": acc_r / MC_iter}, step=epi_iter)
            if (epi_iter % 100 == 0):
                print('Train expert, Episode', epi_iter, 'average reward', acc_r / MC_iter)
            if done:
                expert.train(a1_list, pi_a1_list, r_list)
        expert.save_model()
    else:
        expert.load_model()

    # record expert policy
    exp_s_list = []
    exp_a_list = []
    expert_frames = list()
    state = env.reset()
    for MC_iter in range(args.max_MC_iter):
        action1, pi_a1, log_prob1 = expert.get_action(state)
        exp_s_list.append(state)
        exp_a_list.append(action1)
        expert_frames.append(env.render())
        next_state, reward, done, _ = env.step([action1, 0])
        state = next_state
        if (epi_iter % 100 == 0):
            print('step', MC_iter, 'expert 1 at', exp_s_list[MC_iter], 'expert 1 action', exp_a_list[MC_iter], 'reward', reward, 'done', done)
        if done:
            break

    # generative adversarial imitation learning from [exp_s_list, exp_a_list]
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
                acc_r = acc_r + reward
                r1_list.append(reward)
                if done:
                    break
            if args.wandb:
                wandb.log(data={"agent/train_ep_return": acc_r / MC_iter}, step=epi_iter)
            if (epi_iter % 100 == 0):
                print('Imitate by GAIL, Episode', epi_iter, 'average reward', acc_r / MC_iter)
            # train Discriminator
            agent.train_D(s1_list, a1_list, exp_s_list, exp_a_list)

            # train Generator
            agent.train_G(s1_list, a1_list, log_pi_a1_list, r1_list, exp_s_list, exp_a_list)
        agent.save_model()
    else:
        agent.load_model()

    # learnt policy
    print('expert trajectory')
    # for i in range(len(exp_a_list)):
    #     print('step', i, 'agent 1 at', exp_s_list[i], 'agent 1 action', exp_a_list[i])
    import imageio
    imageio.mimsave("expert.gif", expert_frames)
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
    imageio.mimsave("agent.gif", frames)
    wandb.log({"video/agent": wandb.Video("agent.gif", fps=4, format="gif")}, step=args.max_epi_iter)
