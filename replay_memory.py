import random
import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayMemory:
    def __init__(self, capacity=1000000, seed=123456, state_only=False):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.device = device
        self.state_only = state_only

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        #self.buffer[self.position] = (state, action, reward, next_state, done)
        self.buffer[self.position] = [state, action, reward, next_state, done]
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def get_full_state_memory(self):
        full_state_memory, _, _, _, _ = map(np.stack, zip(*self.buffer))
        return full_state_memory

    def __len__(self):
        return len(self.buffer)
    
    def relabel_memory(self, reward_network):
        # for i in range(len(self.buffer)):
        #     self.buffer[i][2] = reward_network.get_reward(self.buffer[i][0], self.buffer[i][1]).detach().cpu().numpy()[0]
            
        
        ##################################
        if self.state_only:
            traj_list = [traj[0] for traj in self.buffer]
        else:
            traj_list = [np.concatenate([traj[0],traj[1]], axis=-1) for traj in self.buffer]
        traj_list = np.array(traj_list)
        with torch.no_grad():
            rewards = reward_network.reward_network(torch.from_numpy(traj_list).float().to(self.device))
        rewards = np.array(rewards.cpu())
        
        for i in range(len(self.buffer)):
            self.buffer[i][2] = rewards[i][0]
            
