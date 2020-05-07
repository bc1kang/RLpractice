# PPO
# Last update: 2020/05/07

import gym
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

learning_rate = 0.0005
gamma         = 0.98
lamda         = 0.95
eps_clip      = 0.1
epoch         = 3

class PPO(nn.Module):
    def __init__(self):
        super().__init__()
        self.data = []

        self.fc1 = nn.Linear(4,256)
        self.fc_pi = nn.Linear(256,2)
        self.fc_v = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, ndim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim = ndim)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def put_data(self, newdata):
        self.data.append(newdata)

    def make_batch(self):
        obs_list, action_list, reward_list, obs_next_list, prob_list, done_list = [], [], [], [], [], []
        for newdata in self.data:
            obs, action, reward, obs_next, prob, done = newdata
            
            obs_list.append(obs)
            action_list.append([action])
            reward_list.append([reward])
            obs_next_list.append(obs_next)
            prob_list.append([prob])
            done_list.append([0 if done else 1])
        
        obs      = torch.tensor(obs_list, dtype = torch.float)
        action   = torch.tensor(action_list)
        reward   = torch.tensor(reward_list)
        obs_next = torch.tensor(obs_next_list, dtype = torch.float)
        prob     = torch.tensor(prob_list)
        done     = torch.tensor(done_list, dtype = torch.float)
        
        self.data = []
        return obs, action, reward, obs_next, prob, done

    def train_net(self):
        obs, action, reward, obs_next, prob, done = self.make_batch()

        for _ in range(epoch):
            TD_target = reward + gamma * self.v(obs_next) * done
            delta = TD_target - self.v(obs)
            delta = delta.detach().numpy()

            adv_list = []
            adv = 0.0
            for delta_t in delta[::-1]:
                adv = gamma * lamda * adv + delta_t[0]
                adv_list.append([adv])
            adv_list.reverse()
            adv = torch.tensor(adv_list, dtype = torch.float)

            pi = self.pi(obs, ndim = 1)
            pi_a = pi.gather(1,action)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob))

            temp1 = ratio * adv
            temp2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * adv
            loss = -torch.min(temp1, temp2) + F.smooth_l1_loss(self.v(obs), TD_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()


def main():
    env = gym.make('CartPole-v1')
    model = PPO()
    score = 0.0
    print_interval = 20

    for i_episode in range(10000):
        obs = env.reset()
        done = False
        while not done:
            for t in range(20):
                prob = model.pi(torch.from_numpy(obs).float())
                index = Categorical(prob)
                action = index.sample().item()
                obs_next, reward, done, info = env.step(action)
                model.put_data((obs, action, reward/100.0, obs_next, prob[action].item(), done))
                obs = obs_next

                score += reward

                if score/print_interval >= 400:
                    env.render()
                    time.sleep(0.01)

                if done:
                    #print("Episode finished after {} timesteps".format(t+1))
                    break
            
            model.train_net()
        if i_episode % print_interval == 0 and i_episode != 0:
            print("# of episode: {}, avg score: {:.1f}".format(i_episode, score/print_interval))
            score = 0.0

    env.close()

if __name__ == '__main__':
    main()
