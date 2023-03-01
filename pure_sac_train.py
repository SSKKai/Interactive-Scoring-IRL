import argparse
import time
from collections import deque
import numpy as np
import itertools
import torch
from sac import SAC
from utils import get_wandb_config, set_seeds
# from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
from reward_net import RewardNetwork
import glfw
import proplot as pplt

from sklearn.cluster import KMeans

# import env
import gym  # mujoco
from custom_env import CustomEnv  # RLBench
import metaworld.envs.mujoco.env_dict as _env_dict  # metaworld

# import config and logger
import hydra
import wandb


class OPRRL(object):
    def __init__(self, config):

        # Configurations
        self.sac_hyparams = config.sac
        self.reward_hyparams = config.reward
        self.env_config = config.env

        # Experiment setup
        self.episode_len = config.experiment.episode_len
        self.max_episodes = config.experiment.max_episodes
        self.seeds = config.experiment.seed
        self.change_flag_reward = config.experiment.change_flag_reward

        # Environment
        self.env_type = config.experiment.env_type

        if self.env_type == "rlbench":
            self.env = CustomEnv(self.env_config)
            self.env.reset()

        elif self.env_type == "mujoco":
            if self.env_config.terminate_when_unhealthy is None:
                self.env = gym.make(self.env_config.task)
            else:
                self.env = gym.make(self.env_config.task, terminate_when_unhealthy=self.env_config.terminate_when_unhealthy)
            self.env._max_episode_steps = self.episode_len
            self.env.seed(self.seeds)
            self.env.action_space.seed(self.seeds)

        elif self.env_type == 'metaworld':
            env_cls = _env_dict.ALL_V2_ENVIRONMENTS[self.env_config.task]
            self.env = env_cls()
            self.env._freeze_rand_vec = False
            self.env._set_task_called = True
            self.env.seed(self.seeds)
            self.env.action_space.seed(self.seeds)
            self.env.max_path_length = self.episode_len

        else:
            raise Exception('wrong environment type, available: rlbench/mujoco/metaworld')

        set_seeds(self.seeds)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space

        
        # Agent
        self.agent = SAC(self.state_dim, self.action_dim, args=self.sac_hyparams)
        
        # Memory
        self.agent_memory = ReplayMemory(self.sac_hyparams.replay_size, self.seeds, False)

        # Reward Net
        self.reward_network = RewardNetwork(self.state_dim, self.action_dim.shape[0], self.episode_len, self.env_type, args=self.reward_hyparams)

        # wandb logger
        self.wandb_log = config.experiment.wandb_log
        if self.wandb_log:
            config_wandb = get_wandb_config(config)
            self.logger = wandb.init(config = config, project='oprrl_'+config.experiment.env_type+'_'+config.env.task+'_experiment')
            self.logger.config.update(config_wandb)
        
        # Training Loop
        self.total_numsteps = 0
        self.updates = 0
        self.rank_count = 0
        self.rank_num = 0

        self.start_episodes = self.sac_hyparams.start_episodes
        self.pretrain_episodes = self.sac_hyparams.pretrain_episodes
        
        self.reward_list = []
        self.e_reward_list = []
        self.episode_len_list = []

        self.traj_buffer = []
        self.new_trajectory = True

        
    def evaluate(self, i_episode=20, episode_len=250, evaluate_mode=True):
        print("----------------------------------------")
        total_reward = []
        for _ in range(i_episode):
            state = self.env.reset()
            episode_reward = 0
            done = False
            episode_steps = 0
            while not done:
                if self.env_type == "mujoco" or self.env_type == 'metaworld':
                    if self.env_config.render:
                        self.env.render()
                action = self.agent.select_action(state, evaluate=evaluate_mode)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                state = next_state
                episode_steps += 1
                if episode_steps == episode_len:
                    done = True
            print("Reward: {}".format(round(episode_reward, 2)))
            total_reward.append(episode_reward)

        print("----------------------------------------")
        return np.array(total_reward)

    def reward_analysis(self, episode_len, evaluate_mode=True):
        print("----------------------------------------")
        reward_t_ls = []
        reward_p_ls = []
        episode_reward = 0
        episode_reward_prime = 0

        state = self.env.reset()
        done = False
        episode_steps = 0
        while not done:
            if self.env_type == "mujoco" or self.env_type == 'metaworld':
                if self.env_config.render:
                    self.env.render()
            action = self.agent.select_action(state, evaluate=evaluate_mode)
            next_state, reward, done, _ = self.env.step(action)
            reward_prime = self.reward_network.get_reward(state, action).detach().cpu().numpy()[0]

            reward_t_ls.append(reward)
            reward_p_ls.append(reward_prime)

            state = next_state
            episode_steps += 1
            episode_reward += reward
            episode_reward_prime += reward_prime
            if episode_steps == episode_len:
                done = True

        print(episode_reward, episode_reward_prime)
        print("----------------------------------------")

        reward_data = np.stack((np.array(reward_p_ls)*100, np.array(reward_t_ls))).T

        fig = pplt.figure(suptitle='Single subplot')
        ax = fig.subplot(xlabel='x axis', ylabel='y axis')
        ax.plot(reward_data, lw=2)
        fig.show()

        return reward_data, fig

    def reward_analysis_episodic(self, num_to_sample):
        all_rewards = np.array(self.e_reward_list).reshape(-1,1)
        kmeans = KMeans(n_clusters=num_to_sample, random_state=0).fit(all_rewards)
        centers = kmeans.cluster_centers_.squeeze(axis=-1)
        sample_index = []
        for center in centers:
            idx = (np.abs(all_rewards - center)).argmin()
            sample_index.append(idx)

        rewards_t = [self.e_reward_list[i] for i in sample_index]
        rewards_t = np.array(rewards_t)

        trajs = [self.traj_buffer[i] for i in sample_index]
        trajs = np.array(trajs)
        trajs = torch.from_numpy(trajs).float().to(self.reward_network.device)
        rewards_p = self.reward_network.reward_network(trajs)
        rewards_p = torch.squeeze(rewards_p, axis=-1)
        rewards_p = rewards_p.sum(axis=1)
        rewards_p = rewards_p.detach().cpu().numpy()

        return rewards_p, rewards_t

    def push_data(self, state, action, reward, done):
        inputs = np.concatenate([state, action], axis=-1)

        if self.new_trajectory:
            self.traj_buffer.append([])
            self.new_trajectory = False

        self.traj_buffer[-1].append(inputs)
        if done:
            self.new_trajectory = True

    def train(self):

        succ_de = deque(maxlen=50)
        print_flag = 0
        model_num = 1

        for self.i_episode in itertools.count(1):
            episode_reward = 0
            episode_reward_prime = 0
            episode_steps = 0
            done = False
            state = self.env.reset()

            episode_flag = 0
            episode_succ = False

            #############################################################################
            ############################ step session start #############################
            #############################################################################
            while not done:
                if self.i_episode <= self.start_episodes:
                    if self.env_type == "rlbench":
                        action = self.env.randn_action()  # Sample random action
                    elif self.env_type == "mujoco" or self.env_type == 'metaworld':
                        action = self.env.action_space.sample()
                else:
                    action = self.agent.select_action(state)  # Sample action from policy
        
                if len(self.agent_memory) > self.sac_hyparams.batch_size:
                    # Number of updates per step in environment
                    for i in range(self.sac_hyparams.updates_per_step):
                        # Update parameters of all the networks
                        if self.i_episode > self.start_episodes + self.pretrain_episodes or self.pretrain_episodes == 0:
                            if print_flag == 0:
                                print('train session start')
                                print_flag += 1
                            critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = self.agent.update_parameters(self.agent_memory, self.sac_hyparams.batch_size, self.updates)
                        else:
                            critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = self.agent.update_parameters_pretrain(self.agent_memory, self.sac_hyparams.batch_size, self.updates)
                        self.updates += 1
                        if self.wandb_log:
                            self.logger.log({"critic_1_loss": critic_1_loss, "critic_2_loss": critic_2_loss, "policy_loss": policy_loss})

                next_state, reward, done, info = self.env.step(action)

                if self.env_type == 'metaworld' and episode_flag == 0:
                    if info['success']:
                        episode_succ = True
                        episode_flag = 1
                
                episode_steps += 1
                self.total_numsteps += 1
                episode_reward += reward

                # self.env.render()
        
                # Ignore the "done" signal if it comes from hitting the time horizon.
                # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
                mask = 1 if episode_steps == self.episode_len else float(not done)

                if episode_steps % self.episode_len == 0:
                    done = True

                self.agent_memory.push(state, action, reward, next_state, mask)
                self.push_data(state, action, reward, done)# push data to replay buffer

                state = next_state
            #############################################################################
            ############################## step session end #############################
            #############################################################################

            if self.wandb_log:
                if self.env_type == "mujoco":
                    self.logger.log({"e_reward": episode_reward, "episode_steps": episode_steps, "i_episode": self.i_episode})
                else:
                    if self.env_type == "rlbench":
                        task_succ = True if episode_steps < self.episode_len else False

                    elif self.env_type == 'metaworld':
                        task_succ = episode_succ

                    succ_de.append(task_succ)
                    succ_rate = np.mean(succ_de)
                    self.logger.log({"e_reward": episode_reward, "episode_steps": episode_steps, "success_rate": succ_rate, "i_episode": self.i_episode})

            self.e_reward_list.append(episode_reward)
            self.episode_len_list.append(episode_steps)
            print("E{}, t numsteps: {}, e steps: {}, reward: {}".format(self.i_episode, self.total_numsteps, episode_steps, round(episode_reward, 2)))


            if self.i_episode == self.start_episodes + self.pretrain_episodes and self.pretrain_episodes > 0:
                self.agent.reset_critic()
                self.agent.reset_actor()
                for _ in range(100):
                    self.agent.update_parameters(self.agent_memory, self.sac_hyparams.batch_size, self.updates)
                print('pretrain session end')

            if self.i_episode % self.sac_hyparams.eval_per_episode == 0 and self.sac_hyparams.eval is True:
                self.evaluate(self.sac_hyparams.eval_episodes, self.episode_len)

            # for reward analysis experiment
            # if self.i_episode % 200 == 0:
            #     self.agent.save_model('bp_ra',str(model_num))
            #     model_num += 1
            
            if self.i_episode >= self.max_episodes - 1:
                break
        # self.env.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config-name', default="mujoco-HalfCheetah-fb250",
                        help='Please specify the config name (default: mujoco-HalfCheetah-fb250)')
    args = parser.parse_args()

    with hydra.initialize(config_path="config"):
        config = hydra.compose(config_name=args.config_name)
    print(config.experiment.description)

    oprrl = OPRRL(config)
    oprrl.train()


