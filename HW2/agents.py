import random
import numpy as np
import math
from collections import defaultdict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from dqn_utils import *


def get_empty_cells(state):
    return [i for i,x in enumerate(state) if x == '1']

def get_used_cells(state):
    return [i for i,x in enumerate(state) if x != '1']


class RandomAgent:
    "The simpliest agent, just return random empty cell"
    def select_best_action(self, state):
        return random.choice(get_empty_cells(state))
    

class TQAgent:
    "Tabular Q-learning implementation"
    def __init__(self, num_of_states, lr, gamma, eps, side):
        # init Q-table with zeros by default
        self.Q = defaultdict(lambda: np.zeros(num_of_states))
        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.side = side
        self.curr_action, self.curr_state = None, None
    
    def select_best_action(self, state):
        # greedy
        self.Q[state][get_used_cells(state)] = -np.inf
        return np.argmax(self.Q[state])
    
    def select_action(self, state):
        # epsilon-greedy
        if np.random.rand() < self.eps:
            return random.choice(get_empty_cells(state))
        return self.select_best_action(state)
    
    def update_Q(self, next_state, next_action, reward):
        # update Q-value starting from the second step
        if self.curr_action: # and self.curr_state
            self.Q[self.curr_state][self.curr_action] += self.lr * \
            (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[self.curr_state][self.curr_action])
        self.curr_state = next_state
        self.curr_action = next_action
        

class DQNAgent:
    "Deep Q-Net agent"
    def __init__(self, model, hid_size, game_size, action_size, mem_size, batch_size,
                 lr, gamma, epsilon, update_every, target_update_every, side):
#         super(DQNAgent, self).__init__()
        self.game_size = game_size
        self.gamma = gamma
        self.eps = epsilon
        self.memory = ReplayMemory(mem_size)
        self.side = side
        self.batch_size = batch_size
        self.update_every = update_every
        self.target_update_every = target_update_every
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.qnet_local = model(hid_size, game_size, action_size).to(self.device)
        self.qnet_target = model(hid_size, game_size, action_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.qnet_local.parameters(), lr=lr)
        self.curr_action, self.curr_state = None, None
        self.t_step = 0 # time tick for update checks every step
        print(f'{side}-agent. Using device: {self.device}')
        
    @staticmethod
    def format_states(state, size):
        return np.array(list(state)).astype(float).reshape(size, size)
    
    def update_memory(self, next_state, next_action, reward):
        if self.curr_action:
            self.memory.store(
                (self.format_states(self.curr_state, self.game_size), self.curr_action,
                 self.format_states(next_state, self.game_size), reward)
            )
        self.curr_state, self.curr_action = next_state, next_action

    def step(self, state, action, reward):
        self.update_memory(state, action, reward)
        self.t_step += 1
        if (self.t_step % self.update_every) == 0 and len(self.memory) > self.batch_size:
            # learn on random samples
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)
        if (self.t_step % self.target_update_every) == 0:
            # save local QNet weights to Target QNet
            self.qnet_target.load_state_dict(deepcopy(self.qnet_local.state_dict()))
            # print(self.compare_models())
    
    def learn(self, experiences):
        # training step, update DQN by experiences batch
        exp = Experience(*zip(*experiences))
        # better move this unpacking to memory.SAMPLE() in future
        states = torch.Tensor(exp.state).view(-1, 1, self.game_size, self.game_size).to(self.device)
        actions = torch.Tensor(exp.action).unsqueeze(1).long().to(self.device)
        rewards = torch.Tensor(exp.reward).unsqueeze(1).to(self.device)
        next_states = torch.Tensor(exp.next_state).view(-1, 1, self.game_size, self.game_size).to(self.device)
        
        q_expected = self.qnet_local(states).gather(1, actions)
        q_targets_next = self.qnet_target(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + self.gamma * q_targets_next
        
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()        
    
    def select_best_action(self, state):
        # greedy
        state_tensor = torch.Tensor(self.format_states(state, self.game_size)).view(-1, 1, self.game_size, self.game_size).to(self.device)
        actions = self.qnet_target(state_tensor).flatten()
        actions[get_used_cells(state)] = -np.inf 
        return actions.argmax().item()
    
    def select_action(self, state):
        # epsilon-greedy
        if np.random.rand() < self.eps:
            return random.choice(get_empty_cells(state))
        return self.select_best_action(state)

        
class RolloutAgent:
    "Rollouts implementation (vs random choices)"
    def __init__(self, env, n_rollouts, side):
        self.env = env
        self.opp_agent = RandomAgent()
        self.side = side
        self.n_rollouts = n_rollouts
        
    def make_rollout(self, env):
        roll_env = deepcopy(env)
        state, empty_cells, turn = roll_env.getState()
        roll_side = -turn
        done = False
        while not done:
            action = self.opp_agent.select_best_action(state)
            (state, empty_cells, turn), reward, done, _ = roll_env.step(env.action_from_int(action))
        return reward * roll_side

    def eval_by_rollouts(self):
        state, empty_cells, turn = self.env.getState()
        curr_side = turn
        action_estimates = {}
        for action in get_empty_cells(state):
            curr_env = deepcopy(self.env)
            (state, empty_cells, turn), reward, done, _ = curr_env.step(curr_env.action_from_int(action))
            if done:
                action_estimates[action] = reward * curr_side
            else:
                action_estimates[action] = np.mean([self.make_rollout(curr_env) for _ in range(self.n_rollouts)])
        
        return max(action_estimates, key=action_estimates.get)
    
    def run_episode(self):
        self.env.reset()
        state, empty_cells, turn = self.env.getState()
        correction = 1 if self.side == 'x' else -1
        done = False
        while not done:
            if self.side == 'x':
                action = self.eval_by_rollouts() if turn == 1 \
                else self.opp_agent.select_best_action(state)
            else:
                action = self.eval_by_rollouts() if turn == -1 \
                else self.opp_agent.select_best_action(state)
            (state, empty_cells, turn), reward, done, _ = self.env.step(self.env.action_from_int(action))
            
        return reward * correction
    

class MCTSAgent:
    "Monte Carlo Tree Search"
    def __init__(self, env, side='x', exploration_weight=1):
        self.env = env
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.C = exploration_weight
        self.side = 1 if side == 'x' else -1
        
    def select_action(self, node1, node2):
        return np.argwhere(
            np.array(list(node1)).reshape(self.env.n_rows, self.env.n_cols) 
            != np.array(list(node2)).reshape(self.env.n_rows, self.env.n_cols)
        ).flatten()
    
    def uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"
        log_N_vertex = math.log(self.N[node])
        uct = []
        for child in self.children[node]:
            uct.append(self.Q[child] / self.N[child] + self.C * math.sqrt(log_N_vertex / self.N[child]))
        return self.children[node][np.argmax(uct)]
    
    def select_node(self, node):
        "Find an unexplored descendent of `node`"
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                return path
            for child in self.children[node]:
                if child not in self.N:
                    # select first unexplored child
                    action = self.select_action(node, child)
                    path.append(child)
                    self.env.step(action)
                    return path
            if self.side == self.env.curTurn:
                # continue selecting by UCT
                child = self.uct_select(node)
                action = self.select_action(node, child)
            else:
                # opposite agent makes random action
                action = random.choice(self.env.getEmptySpaces())
            (node, _, _), _, _, _ = self.env.step(action)
    
    def expand(self):
        "Update the `children` dict with the children of `node`"
        node, empty_cells, turn = self.env.getState()
        if node in self.children:
            return  # already expanded
        children = []
        if not self.env.gameOver:
            for action in empty_cells:
                self.env.makeMove(turn, action[0], action[1])
                children.append(self.env.getHash())
                self.env.makeMove(0, action[0], action[1])

        self.children[node] = children
    
    def backprop(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
            reward = 1 - reward # inverse for enemy
    
    def make_rollout(self):
        if self.env.gameOver:
            self.env.curTurn = -self.env.curTurn
            reward = self.env.isTerminal()
            return int(reward * self.env.curTurn > 0), True
        
        env = deepcopy(self.env)
        _, empty_cells, turn = env.getState()
        done = False
        while not done:
            action = random.choice(empty_cells)
            (_, empty_cells, _), reward, done, _ = env.step(action)
        
        fin_reward = None
        if turn == 1:
            fin_reward = 0 if reward >= 0 else 1
        elif turn == -1:
            fin_reward = 0 if reward <= 0 else 1
        return fin_reward, False

    def select_best_action(self):
        "Greedy action selection"
        node, empty_cells, _ = self.env.getState()
        
        if node not in self.children:
            return random.choice(empty_cells)
        
        # get mean Q for each available action
        q_vals = []
#         print('select from', self.children[node])
        for child in self.children[node]:
            if child not in self.N:
                q_vals.append(-np.inf)
            else:
                q_vals.append(self.Q[child] / self.N[child])
        return empty_cells[np.argmax(q_vals)]
    
    def learn(self, n_episodes):
        "Learn for n_episodes"
        for _ in range(n_episodes):
            self.env.reset()
            done = False
            path = []
            while not done:
                node, _, _ = self.env.getState()
                curr_path = self.select_node(node)
                path += curr_path
                self.expand()
                reward, done = self.make_rollout()
                self.backprop(path, reward)
                path = path[:-1]
    
    def validate(self, n_episodes):
        "Validate vs random policy for n_episodes"
        all_rewards = []
        for _ in range(n_episodes):
            self.env.reset()
            state, empty_cells, turn = self.env.getState()
            done = False
            while not done:
                if (turn == 1 and self.side == 1) or (turn == -1 and self.side == -1):
                    action = self.select_best_action()
                else:
                    action = random.choice(empty_cells)
                
                (state, empty_cells, turn), reward, done, _ = self.env.step(action)
            if reward == -10:
                print(state, empty_cells)
                raise ValueError('Incorrect action')
            fin_reward = reward if self.side == 1 else -reward
            all_rewards.append(fin_reward)
        return all_rewards
