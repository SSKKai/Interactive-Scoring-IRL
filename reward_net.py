import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.distributions import Beta,Normal
from sklearn.cluster import KMeans
from scipy.stats import kendalltau
import math
import random
import heapq
import similaritymeasures
from model import TorchRunningMeanStd



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_state_entropy(state_batch, full_state_memory, k):
    batch_size = 500
    with torch.no_grad():
        dists = []
        for idx in range(len(full_state_memory) // batch_size + 1):
            start = idx * batch_size
            end = (idx + 1) * batch_size
            dist = torch.norm(
                state_batch[:, None, :] - full_state_memory[None, start:end, :], dim=-1, p=2
            )
            dists.append(dist)

        dists = torch.cat(dists, dim=1)
        knn_dists = torch.kthvalue(dists, k=k + 1, dim=1).values
        state_entropy = knn_dists
    return state_entropy.unsqueeze(1)


class Reward_Net(nn.Module):
    def __init__(self, num_inputs, hidden_dim, negative_network_output):
        super(Reward_Net, self).__init__()
        self.negative_network_output = negative_network_output

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)
        
    def forward(self, inputs):
        x = F.leaky_relu(self.linear1(inputs))
        x = F.leaky_relu(self.linear2(x))
        x = F.leaky_relu(self.linear3(x))
        x = torch.tanh(self.linear4(x))

        if self.negative_network_output:
            return (x-1)/2
        else:
            return x


class RewardNetwork(object):
    def __init__(self, state_dim, action_dim, episode_length, env_type, args):
        self.device = device
        self.env_type = env_type
        
        self.state_only = args.state_only
        self.new_trajectory = True
        self.new_reward_traj = True
        
        
        self.traj_capacity = args.traj_capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.episode_length = episode_length
        
        self.num_to_rank = args.num_to_rank
        self.rank_noise = args.rank_noise
        self.half_precision = args.half_precision
        self.negative_network_output = args.negative_network_output
        
        if self.state_only:
            num_inputs = self.state_dim
        else:
            num_inputs = self.state_dim+self.action_dim
        self.reward_network = Reward_Net(num_inputs, args.hidden_dim, self.negative_network_output).to(device=self.device)
        
        self.sample_method = args.sample_method
        self.padding_mask_method = args.padding_mask_method
        self.label_type = args.label_type
        self.make_batch_method = args.make_batch_method

        self.buffer = []
        self.index_buffer = []
        self.true_reward_buffer = []
        self.ranked_trajs = []
        self.cartesian_ranked_trajs = []
        self.ranked_labels = []
        self.ranked_labels_true = []
        self.ranked_lens = []
        self.ranked_states = torch.FloatTensor(np.array([])).to(device)
        self.ranked_index = []

        self.priority_alpha = args.priority_alpha
        self.entropy_alpha = args.entropy_alpha
        self.new_bonus = args.new_bonus
        self.sample_batch_priority = np.array([])
        self.sample_batch_entropy = np.array([])

        self.equal_threshold = args.equal

        
        self.train_batch_size = 128
        self.epoch = 3
        self.optimizer = optim.Adam(self.reward_network.parameters(), lr=args.lr)

        self.s_ent_stats = TorchRunningMeanStd(shape=[1], device=self.device)

        self.test_flag = False
        self.selected_to_rank_trajs = []
    
    def get_reward(self, state, action):
        if self.state_only:
            inputs = state
        else:
            inputs = np.concatenate([state, action], axis=-1)
        reward = self.reward_network(torch.from_numpy(inputs).float().to(self.device))
        return reward

    def get_batch_entropy(self, traj_batch):
        # full_states = self.ranked_states.copy()
        # full_states = torch.FloatTensor(np.array(full_states)).to(device)

        batch_entropy = []

        for traj in traj_batch:
            state_traj = [sa[:self.state_dim] for sa in traj]
            state_traj = torch.FloatTensor(np.array(state_traj)).to(device)
            state_entropy = compute_state_entropy(state_traj, self.ranked_states, 10)

            #######################################################
            # self.s_ent_stats.update(state_entropy)
            # state_entropy = state_entropy / self.s_ent_stats.std
            #######################################################

            entropy = state_entropy.sum().cpu().numpy().tolist()
            batch_entropy.append(entropy)
        return np.array(batch_entropy)
    
    def push_data(self, state, action, reward, done, i_episode):
        if self.state_only:
            inputs = state
        else:
            inputs = np.concatenate([state, action], axis=-1)
        
        if self.new_trajectory:
            self.buffer.append([])
            self.new_trajectory = False
        
        self.buffer[-1].append(inputs)
        if done:
            self.new_trajectory = True
            self.index_buffer.append(i_episode)
            if len(self.buffer) > self.traj_capacity:
                self.buffer = self.buffer[1:]
                self.index_buffer = self.index_buffer[1:]

        self.add_true_reward(reward, done)
    
    def add_true_reward (self, reward, done):
        if self.new_reward_traj:
            self.true_reward_buffer.append([])
            self.new_reward_traj = False
        
        self.true_reward_buffer[-1].append(reward)
        if done:
            self.new_reward_traj = True
            if len(self.true_reward_buffer) > self.traj_capacity:
                self.true_reward_buffer = self.true_reward_buffer[1:]

    def auto_get_label(self, rank_index):
        rank_label = []
        rank_label_true = []
        for idx in rank_index:
            total_reward = sum(self.true_reward_buffer[idx])
            rank_base = total_reward/self.episode_length

            ##################### rlbench #########################
            if self.env_type == 'rlbench':
                if rank_base <= -3.2:  # closemicrowave/pushbutton vertically: (-)3.2 / pushbutton horizontally: (-)1.8
                    rank = 0
                else:
                    rank = (rank_base/1.8 + 1)*10
                rank_label_true.append(rank)

            ##################### mujoco #########################
            elif self.env_type == 'mujoco':
                if rank_base <= -1.5:
                    rank = 0
                elif -1.5 < rank_base <= 0:
                    rank = 2*(rank_base+1.5)/1.5
                elif total_reward > 0:
                    rank = 2 + 8*rank_base/5
                rank_label_true.append(rank)

                # add score noise
                if self.rank_noise > 0:
                    rank = rank + np.random.normal(0, self.rank_noise)
                # .5 precision
                if self.half_precision:
                    if abs(rank - np.round(rank)) < 0.25:
                        rank = np.round(rank)
                    else:
                        rank = np.floor(rank) + 0.5
                    # limit
                    # rank = max(rank, 0)
                    # rank = min(rank, 10)


            ##################### metaworld #########################
            elif self.env_type == 'metaworld':
                if rank_base <= -1.5:
                    rank = 0
                elif -1.5 < rank_base <= 0:
                    rank = 2*(rank_base+1.5)/1.5
                elif total_reward > 0:
                    rank = 2 + 8*rank_base/5
                rank_label_true.append(rank)

                # add score noise
                if self.rank_noise > 0:
                    rank = rank + np.random.normal(0, self.rank_noise)
                # .5 precision
                if self.half_precision:
                    if abs(rank - np.round(rank)) < 0.25:
                        rank = np.round(rank)
                    else:
                        rank = np.floor(rank) + 0.5
                    # limit
                    # rank = max(rank, 2)
                    # rank = min(rank, 13)
            #######################################

            rank_label.append(rank)
        return rank_label, rank_label_true

    def manual_get_reference(self, rank_index):
        rank_references = []
        _, rank_label_true = self.auto_get_label(rank_index)
        if len(self.ranked_labels) > 0:
            distance_reference_trajs = self.cartesian_ranked_trajs[-100:]
            ranked_predicted_rewards = self.reward_network(torch.from_numpy(np.array(self.ranked_trajs)).float().to(self.device)).cpu().squeeze(dim=-1).sum(axis=1)
        for n,idx in enumerate(rank_index):
            rank_traj_number = self.index_buffer[idx]

            if len(self.ranked_labels) > 0:
                traj_need_to_rank = np.array(self.buffer[idx])[:,0:3]
                traj_distance_ls = [similaritymeasures.dtw(traj_need_to_rank, traj)[0] for traj in distance_reference_trajs]
                min_distance_index = list(map(traj_distance_ls.index, heapq.nsmallest(2,traj_distance_ls)))
                if len(self.cartesian_ranked_trajs) > len(distance_reference_trajs):
                    min_distance_index = [index+len(self.cartesian_ranked_trajs)-100 for index in min_distance_index]

                rank_traj_predicted_reward = self.reward_network(torch.from_numpy(np.array(self.buffer[idx])).float().to(self.device)).cpu().squeeze(dim=-1).sum()
                reward_distance_ls = [abs(ranked_reward-rank_traj_predicted_reward) for ranked_reward in ranked_predicted_rewards]
                min_reward_index = list(map(reward_distance_ls.index, heapq.nsmallest(2,reward_distance_ls)))

                min_index = min_distance_index + min_reward_index
                min_index = [i for n, i in enumerate(min_index) if i not in min_index[:n]]
                ref_labels_ls = [self.ranked_labels[i] for i in min_index]
                ref_traj_number_ls = [self.ranked_index[i] for i in min_index]

            else:
                ref_labels_ls = []
                ref_traj_number_ls = []

            rank_references.append([rank_traj_number,ref_traj_number_ls,ref_labels_ls,rank_label_true[n]])

        return rank_references
    
    def get_trajs_to_rank(self, num_to_rank=None):

        if num_to_rank is None:
            num_to_rank = self.num_to_rank

        if len(self.buffer) < num_to_rank:
            num_to_rank = len(self.buffer)

        if self.sample_method == 'random sample':
            self.selected_to_rank_trajs, rank_index = self.random_sample_buffer(num_to_rank)
        elif self.sample_method == 'distance sample':
            self.selected_to_rank_trajs, rank_index = self.distance_sample_buffer(num_to_rank)
        elif self.sample_method == 'priority sample':
            self.selected_to_rank_trajs, rank_index = self.priority_sample_buffer(num_to_rank)
        elif self.sample_method == 'state entropy sample':
            self.selected_to_rank_trajs, rank_index = self.state_entropy_sample_buffer(num_to_rank)
        else:
            raise Exception('wrong sample method for reward learning')

        return rank_index

    def push_ranked_data(self, rank_label, rank_label_true, rank_index, ref_label_change=None, skip_index=None):
        rank_trajs = self.selected_to_rank_trajs
        if skip_index is not None:
            if len(skip_index) > 0:
                rank_index = [index for i, index in enumerate(rank_index) if i not in skip_index]
                rank_trajs = [rank_traj for i, rank_traj in enumerate(rank_trajs) if i not in skip_index]

        if len(rank_index) > 0:
            if 'entropy' in self.sample_method or 'entropy' in self.make_batch_method:
                states = torch.FloatTensor(np.array([sa[:self.state_dim] for traj in rank_trajs for sa in traj])).to(device)
                self.ranked_states = torch.cat((self.ranked_states, states), 0)

            rank_trajs, rank_len = self.padding(rank_trajs)

            self.ranked_trajs.extend(rank_trajs)
            self.ranked_labels.extend(rank_label)
            self.ranked_lens.extend(rank_len)

            self.ranked_index.extend([self.index_buffer[idx] for idx in rank_index])
            self.cartesian_ranked_trajs.extend([np.array(traj)[:, 0:3] for traj in rank_trajs])
            self.ranked_labels_true.extend(rank_label_true)

            self.buffer = [self.buffer[i] for i in range(len(self.buffer)) if (i not in rank_index)]
            self.index_buffer = [self.index_buffer[i] for i in range(len(self.index_buffer)) if (i not in rank_index)]
            self.true_reward_buffer = [self.true_reward_buffer[i] for i in range(len(self.true_reward_buffer)) if (i not in rank_index)]

            if ref_label_change is not None:
                for i in range(len(ref_label_change[0])):
                    change_ranked_index =self.ranked_index.index(ref_label_change[0][i])
                    self.ranked_labels[change_ranked_index] = ref_label_change[1][i]

            if 'entropy' in self.make_batch_method:
                self.update_sample_prob_entropy(self.entropy_alpha, len(rank_index))
            if 'priority' in self.make_batch_method:
                self.update_sample_prob_priority(self.priority_alpha, len(rank_index))
            # self.update_sample_prob_priority(self.priority_alpha, num_to_rank)

        k_tau, _ = kendalltau(self.ranked_labels_true, self.ranked_labels)
        return k_tau


    def random_sample_buffer(self, num_to_rank):

        sample_index = random.sample(range(len(self.buffer)), num_to_rank)
        sample_batch = []
        
        for i in range(num_to_rank):
            sample_batch.append(self.buffer[sample_index[i]])
        # batch = random.sample(self.buffer, self.num_to_rank)
        return sample_batch, sample_index

    def distance_sample_buffer(self, num_to_rank):

        padded_buffer, buffer_len_list = self.padding(self.buffer)
        buffer_mask = self.make_mask(buffer_len_list)[0]

        buffer_mask = buffer_mask.to(self.device)
        padded_buffer = np.array(padded_buffer)

        rewards = self.reward_network(torch.from_numpy(padded_buffer).float().to(self.device))
        rewards = torch.squeeze(rewards, axis=-1)
        rewards = rewards * buffer_mask
        rewards = rewards.sum(axis=1)

        rewards = rewards.detach().cpu().numpy().reshape(-1, 1)
        kmeans = KMeans(n_clusters=num_to_rank, random_state=0).fit(rewards)
        centers = kmeans.cluster_centers_.squeeze(axis=-1)

        sample_index = []
        for center in centers:
            idx = (np.abs(rewards - center)).argmin()
            sample_index.append(idx)

        sample_batch = []
        for i in range(num_to_rank):
            sample_batch.append(self.buffer[sample_index[i]])

        return sample_batch, sample_index

    def priority_sample_buffer(self, num_to_rank):
        num_to_rank_by_priority = int(num_to_rank*2/5)
        padded_buffer, buffer_len_list = self.padding(self.buffer)
        buffer_mask = self.make_mask(buffer_len_list)[0]

        buffer_mask = buffer_mask.to(self.device)
        padded_buffer = np.array(padded_buffer)

        rewards = self.reward_network(torch.from_numpy(padded_buffer).float().to(self.device))
        rewards = torch.squeeze(rewards, axis=-1)
        rewards = rewards * buffer_mask
        rewards = rewards.sum(axis=1)

        rewards_priority = rewards.detach().cpu().numpy().tolist()
        sample_index = list(map(rewards_priority.index, heapq.nlargest(num_to_rank_by_priority, rewards_priority)))


        rewards = rewards.detach().cpu().numpy().reshape(-1, 1)
        kmeans = KMeans(n_clusters=num_to_rank, random_state=0).fit(rewards)
        centers = kmeans.cluster_centers_.squeeze(axis=-1)

        for center in centers:
            idx = (np.abs(rewards - center)).argmin()
            if idx not in sample_index:
                sample_index.append(idx)
                if len(sample_index) == num_to_rank:
                    break

        sample_batch = []
        for i in range(num_to_rank):
            sample_batch.append(self.buffer[sample_index[i]])

        return sample_batch, sample_index

    def state_entropy_sample_buffer(self, num_to_rank):

        ############################################
        # buffer_rewards = [sum(tr) for tr in self.true_reward_buffer]
        # best_index = [buffer_rewards.index(max(buffer_rewards))]
        ############################################

        if len(self.ranked_states) == 0:
            sample_index = random.sample(range(len(self.buffer)), num_to_rank)
        else:
            batch_entropy = self.get_batch_entropy(self.buffer.copy())
            sample_index = (-batch_entropy).argsort()[:num_to_rank]
            sample_index = sample_index.tolist()

        ############################################
        # if best_index[0] not in sample_index:
        #     sample_index = best_index + sample_index
        #     del sample_index[-1]
        ############################################

        sample_batch = []
        for i in range(num_to_rank):
            sample_batch.append(self.buffer[sample_index[i]])

        return sample_batch, sample_index


    def padding(self, traj_list, method=None):

        if method is None:
            method = self.padding_mask_method

        batch_len = len(traj_list)
        traj_len_list = [len(traj) for traj in traj_list]
        pad_len_list = [self.episode_length - traj_len_list[i] for i in range(batch_len)]

        pad_list = []

        for i in range(batch_len):
            if "zeros" in method:
                traj_pad = [np.zeros(self.action_dim + self.state_dim) for _ in range(pad_len_list[i])]
            if "edge" in method:
                traj_pad = [traj_list[i][-1] for _ in range(pad_len_list[i])]
            if "last" in method:
                n = [int(s) for s in method.split() if s.isdigit()][0]
                pad_unit = traj_list[i][-n:]
                traj_pad = [pad_unit for _ in range(int(np.ceil(pad_len_list[i]/n)))]
                traj_pad = [sa for st in traj_pad for sa in st]
                if len(traj_pad) > pad_len_list[i]:
                    del traj_pad[0:len(traj_pad) - pad_len_list[i]]

            pad_list.append(traj_pad)

        padded_traj_list = [traj_list[i] + pad_list[i] for i in range(batch_len)]

        return padded_traj_list, traj_len_list


    def update_sample_prob_priority(self, prio_alpha, num_to_rank):
        priority = np.array(self.ranked_labels.copy())
        priority = priority**prio_alpha/(priority**prio_alpha).sum()

        bonus = np.zeros(len(priority))
        bonus[-num_to_rank:] = self.new_bonus/num_to_rank

        priority = priority * (1-self.new_bonus) + bonus

        self.sample_batch_priority = priority

    def update_sample_prob_entropy(self, entropy_alpha, num_to_rank):

        entropy = np.array(self.get_batch_entropy(self.ranked_trajs))
        entropy = entropy**entropy_alpha/(entropy**entropy_alpha).sum()

        bonus = np.zeros(len(entropy))
        bonus[-num_to_rank:] = self.new_bonus / num_to_rank

        entropy = entropy * (1 - self.new_bonus) + bonus

        self.sample_batch_entropy = entropy


    def make_mask(self, traj_len_list):
        dim = np.array(traj_len_list).ndim

        if dim == 1:
            traj_lens = [traj_len_list]
        elif dim == 2:
            if "normal mask" in self.padding_mask_method:
                len1 = [len[0] for len in traj_len_list]
                len2 = [len[1] for len in traj_len_list]
            elif "shortest mask" in self.padding_mask_method:
                len1 = [min(len) for len in traj_len_list]
                len2 = len1
            elif "no mask" in self.padding_mask_method:
                len1 = [self.episode_length for _ in traj_len_list]
                len2 = len1

            acc_len1 = [len[0] for len in traj_len_list]
            acc_len2 = [len[1] for len in traj_len_list]

            traj_lens = [len1, len2, acc_len1, acc_len2]

        masks = []
        for lens in traj_lens:
            lens = torch.tensor(lens)
            mask = torch.arange(self.episode_length)[None, :] < lens[:, None]
            masks.append(mask)

        return masks


    def get_labels(self, rank_list):

        acc_labels = [0 if rank[0] > rank[1] else 1 for rank in rank_list]

        if self.label_type == "onehot":
            labels = [[1, 0] if rank[0] > rank[1] else [0, 1] for rank in rank_list]

        else:
            labels = []
            for rank in rank_list:
                #####################################
                # if rank[0] == rank[1]:
                if abs(rank[0] - rank[1]) < self.equal_threshold:
                #####################################
                    label_1 = 0.5
                    label_2 = 0.5
                else:
                    if "adaptive" in self.label_type:
                        smoothing_alpha = 1 / ((2 + abs(rank[0] - rank[1])) ** 2)
                    elif "smoothing" in self.label_type:
                        try:
                            smoothing_alpha = float(self.label_type.split()[1])
                        except IndexError:
                            smoothing_alpha = 0.05
                    label_1 = (1 - smoothing_alpha) * (rank[0] > rank[1]) + smoothing_alpha / 2
                    label_2 = (1 - smoothing_alpha) * (rank[0] < rank[1]) + smoothing_alpha / 2

                labels.append([label_1, label_2])

        return labels, acc_labels

    def random_batch_index(self, batch_size):
        index_list = []
        while len(index_list) < batch_size:
            idx = random.sample(range(len(self.ranked_trajs)), 2)
            idx.sort()
            if idx not in index_list:
                index_list.extend([idx])
                # if abs(self.ranked_labels[idx[0]] - self.ranked_labels[idx[1]]) < 1 or random.random() > 0:  # 0.8
                #     index_list.extend([idx])

        return index_list

    def priority_batch_index(self, batch_size):
        index = list(range(len(self.ranked_trajs)))
        index_list = []
        while len(index_list) < batch_size:
            idx = random.choices(index, weights=self.sample_batch_priority, k=2)
            idx.sort()
            if idx not in index_list:
                index_list.extend([idx])

        return index_list

    def entropy_batch_index(self, batch_size):
        index = list(range(len(self.ranked_trajs)))
        index_list = []
        while len(index_list) < batch_size:
            idx = random.choices(index, weights=self.sample_batch_entropy, k=2)
            idx.sort()
            if idx not in index_list:
                index_list.extend([idx])

        return index_list

    def priority_random_batch_index(self, batch_size):
        index = list(range(len(self.ranked_trajs)))
        index_list = []
        while len(index_list) < batch_size:
            idx1 = random.choices(index, weights=self.sample_batch_priority, k=1)
            idx2 = random.sample(index, 1)
            idx = idx1 + idx2
            idx.sort()
            if idx not in index_list:
                index_list.extend([idx])

        return index_list

    def entropy_random_batch_index(self, batch_size):
        index = list(range(len(self.ranked_trajs)))
        index_list = []
        while len(index_list) < batch_size:
            idx1 = random.choices(index, weights=self.sample_batch_entropy, k=1)
            idx2 = random.sample(index, 1)
            idx = idx1 + idx2
            idx.sort()
            if idx not in index_list:
                index_list.extend([idx])

        return index_list

    def priority_entropy_batch_index(self, batch_size):
        index = list(range(len(self.ranked_trajs)))
        index_list = []
        while len(index_list) < batch_size:
            idx1 = random.choices(index, weights=self.sample_batch_priority, k=1)
            idx2 = random.choices(index, weights=self.sample_batch_entropy, k=1)
            idx = idx1 + idx2
            idx.sort()
            if idx not in index_list:
                index_list.extend([idx])

        return index_list

    def difference_batch_index(self, batch_size):
        big_index_list = []
        total_len = len(self.ranked_trajs)
        big_batch_size = int(min(total_len*(total_len-1)/2, batch_size*10))
        if big_batch_size % 2 != 0:
            big_batch_size -= 1

        while len(big_index_list) < big_batch_size:
            idx = random.sample(range(len(self.ranked_trajs)), 2)
            idx.sort()
            if idx not in big_index_list:
                if abs(self.ranked_labels[idx[0]] - self.ranked_labels[idx[1]]) < 1 or random.random() > 0:  # 0.8
                    big_index_list.extend([idx])

        traj_list_1 = []
        traj_list_2 = []
        len_list = []

        for idx in big_index_list:
            traj_list_1.extend([self.ranked_trajs[idx[0]]])
            traj_list_2.extend([self.ranked_trajs[idx[1]]])
            len_list.extend([[self.ranked_lens[idx[0]], self.ranked_lens[idx[1]]]])

        traj_list_1 = np.array(traj_list_1)
        traj_list_2 = np.array(traj_list_2)
        _, _, mask_1, mask_2 = self.make_mask(len_list)

        with torch.no_grad():
            reward_1 = self.reward_network(torch.from_numpy(traj_list_1).float().to(self.device)).cpu()
            reward_2 = self.reward_network(torch.from_numpy(traj_list_2).float().to(self.device)).cpu()
            reward_1 = torch.squeeze(reward_1, axis=-1)
            reward_2 = torch.squeeze(reward_2, axis=-1)
            reward_1 = reward_1 * mask_1
            reward_2 = reward_2 * mask_2
            reward_1 = reward_1.sum(axis=1)
            reward_2 = reward_2.sum(axis=1)
            reward = torch.stack((reward_1, reward_2), axis=1)

        diff = F.softmax(reward, dim=-1) * F.log_softmax(reward, dim=-1)
        diff = diff.sum(axis=-1).abs()

        diff = np.array(diff)

        selected_index = (-diff).argsort()[:batch_size]
        index_list = np.array(big_index_list)[selected_index].tolist()

        return index_list


    def make_batch(self):
        total_len = len(self.ranked_trajs)
        if self.train_batch_size <= total_len*(total_len-1)/2:
            batch_size = self.train_batch_size
        else:
            batch_size = 2 * len(self.ranked_trajs)

        # make index
        if self.make_batch_method == 'random':
            index_list = self.random_batch_index(batch_size)
        elif self.make_batch_method == 'priority':
            index_list = self.priority_batch_index(batch_size)
        elif self.make_batch_method == 'priority random':
            index_list = self.priority_random_batch_index(batch_size)
        elif self.make_batch_method == 'entropy':
            index_list = self.entropy_batch_index(batch_size)
        elif self.make_batch_method == 'entropy random':
            index_list = self.entropy_random_batch_index(batch_size)
        elif self.make_batch_method == 'priority entropy':
            index_list = self.priority_entropy_batch_index(batch_size)
        elif self.make_batch_method == 'difference':
            index_list = self.difference_batch_index(batch_size)
        elif self.make_batch_method == 'hybrid':
            index_list = self.priority_random_batch_index(batch_size)
            self.new_bonus = 0.0004 * len(self.ranked_trajs)
        else:
            raise Exception('wrong make batch method for reward learning')

        traj_list_1 = []
        traj_list_2 = []
        rank_list = []
        len_list = []
        for idx in index_list:
            traj_list_1.extend([self.ranked_trajs[idx[0]]])
            traj_list_2.extend([self.ranked_trajs[idx[1]]])
            rank_list.extend([[self.ranked_labels[idx[0]], self.ranked_labels[idx[1]]]])
            len_list.extend([[self.ranked_lens[idx[0]], self.ranked_lens[idx[1]]]])

        return traj_list_1, traj_list_2, rank_list, len_list, index_list



    def learn_reward_soft(self):

        acc_ls = []
        loss_ls = []

        for epoch in range(self.epoch):
            self.optimizer.zero_grad()

            # make batch
            traj_list_1, traj_list_2, rank_list, len_list, index_list = self.make_batch()

            # make mask
            mask_1, mask_2, acc_mask_1, acc_mask_2 = self.make_mask(len_list)

            # make labels
            labels, acc_labels = self.get_labels(rank_list)


            # training batch to device
            traj_list_1 = np.array(traj_list_1)
            traj_list_2 = np.array(traj_list_2)
            traj_list_1 = torch.from_numpy(traj_list_1).float().to(self.device)
            traj_list_2 = torch.from_numpy(traj_list_2).float().to(self.device)
            mask_1 = mask_1.to(self.device)
            mask_2 = mask_2.to(self.device)
            acc_mask_1 = acc_mask_1.to(self.device)
            acc_mask_2 = acc_mask_2.to(self.device)
            labels = torch.tensor(labels).to(self.device)
            acc_labels = torch.tensor(acc_labels).to(self.device)


            # compute loss
            rewards_1 = self.reward_network(traj_list_1)
            rewards_2 = self.reward_network(traj_list_2)

            rewards_1 = torch.squeeze(rewards_1, axis=-1)
            rewards_2 = torch.squeeze(rewards_2, axis=-1)

            acc_rewards_1 = rewards_1 * acc_mask_1
            acc_rewards_2 = rewards_2 * acc_mask_2
            rewards_1 = rewards_1 * mask_1
            rewards_2 = rewards_2 * mask_2

            rewards_1 = rewards_1.sum(axis=1)
            rewards_2 = rewards_2.sum(axis=1)
            rewards = torch.stack((rewards_1, rewards_2),axis=1)

            log_probs = torch.nn.functional.log_softmax(rewards, dim=1)
            loss = -(labels * log_probs).sum() / rewards.shape[0]

            loss.backward()
            self.optimizer.step()

            # compute acc
            acc_rewards_1 = acc_rewards_1.sum(axis=1)
            acc_rewards_2 = acc_rewards_2.sum(axis=1)
            acc_rewards = torch.stack((acc_rewards_1, acc_rewards_2),axis=1)
            _, predicted_labels = torch.max(acc_rewards.data, 1)
            acc = (predicted_labels == acc_labels).sum().item()/len(predicted_labels)

            acc_ls.append(acc)
        acc = np.mean(acc_ls)

        return acc
    

    
    def save_reward_model(self, env_name, version, reward_path = None):
        if not os.path.exists('reward_models/'):
            os.makedirs('reward_models/')
        
        if reward_path is None:
            reward_path = "reward_models/reward_model_{}_{}".format(env_name, version)
        
        print('Saving reward network to {}'.format(reward_path))
        torch.save(self.reward_network.state_dict(), reward_path)
    
    def load_reward_model(self, reward_path):
        if reward_path is not None:
            print('Loading reward network to {}'.format(reward_path))
            self.reward_network.load_state_dict(torch.load(reward_path))
        else:
            print('fail to load reward network, please enter the reward path')
    
    
    def save_trajs(self, env_name, version, trajs_path=None, labels_path=None, num_to_sample = 20):
        if not os.path.exists('saved_trajs/'):
            os.makedirs('saved_trajs/')
        
        if trajs_path is None:
            trajs_path = "saved_trajs/trajs_{}_{}.npy".format(env_name, version)
        if labels_path is None:
            labels_path = "saved_trajs/labels_{}_{}.npy".format(env_name, version)
        
        index = random.sample(range(len(self.ranked_trajs)), num_to_sample)
        
        save_trajs = [self.ranked_trajs[idx] for idx in index]
        save_labels = [self.ranked_labels[idx] for idx in index]
        save_trajs = np.array(save_trajs)
        save_labels = np.array(save_labels)
        
        label_order = np.argsort(save_labels)
        save_labels.sort()
        save_trajs = save_trajs.take(label_order, axis=0)
        
        np.save(trajs_path, save_trajs)
        np.save(labels_path, save_labels)
    
    def test_reward_model(self, trajs_path, labels_path):
        
        trajs = np.load(trajs_path)
        labels = np.load(labels_path)
        
        rewards = self.reward_network(torch.from_numpy(trajs).float().to(self.device))
        rewards = rewards.sum(axis=1)
        rewards = rewards.detach().cpu().numpy()
        
        return rewards, labels
        
        










