import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import math
import numpy as np

device = torch.device("cpu")  # torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorModel(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(ActorModel, self).__init__()

        hid_size = 128

        self.fc1 = nn.Linear(in_features=obs_dim, out_features=hid_size)
        self.fc2 = nn.Linear(in_features=hid_size, out_features=hid_size)
        self.fc_mean = nn.Linear(in_features=hid_size, out_features=act_dim)
        self.fc_std = nn.Linear(in_features=hid_size, out_features=act_dim)

    def policy(self, obs):
        hid1 = F.relu(self.fc1(obs)).squeeze(0)
        hid2 = F.relu(self.fc2(hid1))
        act_mean = self.fc_mean(hid2)
        act_std = F.softplus(self.fc_std(hid2))
        act_std = torch.clamp(act_std, min=1e-6)
        dist = torch.distributions.Normal(act_mean, act_std)
        sample = dist.rsample()
        action = torch.tanh(sample)

        log_prob = dist.log_prob(sample)
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-7)
        action = (action + 1) / 2
        return action, log_prob, sample

    def predict(self, obs):
        hid1 = F.relu(self.fc1(obs)).squeeze(0)
        hid2 = F.relu(self.fc2(hid1))
        act_mean = self.fc_mean(hid2)
        action = torch.tanh(act_mean)
        action = (action + 1) / 2
        return action


# Critic Model (Value Function)
class CriticModel(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(CriticModel, self).__init__()

        hid_size = 128

        self.fc1 = nn.Linear(in_features=obs_dim + act_dim, out_features=hid_size)  
        self.fc2 = nn.Linear(in_features=hid_size, out_features=hid_size)
        self.fc3 = nn.Linear(in_features=hid_size, out_features=1)

        self.fc4 = nn.Linear(in_features=obs_dim + act_dim, out_features=hid_size)  
        self.fc5 = nn.Linear(in_features=hid_size, out_features=hid_size)
        self.fc6 = nn.Linear(in_features=hid_size, out_features=1)

    def value(self, obs, act):
        concat = torch.cat((obs, act), dim=1)

        hid1 = F.relu(self.fc1(concat))
        hid2 = F.relu(self.fc2(hid1))
        Q1 = self.fc3(hid2)

        hid3 = F.relu(self.fc4(concat))
        hid4 = F.relu(self.fc5(hid3))
        Q2 = self.fc6(hid4)

        return Q1, Q2


class Model(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Model, self).__init__()
        self.critic_model = CriticModel(obs_dim, act_dim)

    def value(self, obs, act):
        return self.critic_model.value(obs, act)

    def get_critic_params(self):
        return self.critic_model.parameters()


class SAC(nn.Module):
    def __init__(self,
                 model,
                 actor_model,
                 gamma=None,
                 tau=None,
                 actor_lr=None,
                 critic_lr=None,
                 policy_noise=0.2,
                 noise_clip=0.5,
                 policy_freq=2):

        super(SAC, self).__init__()

        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0
        self.log_alpha = torch.tensor(math.log(0.01))
        self.log_alpha.requires_grad = True
        self.target_entropy = -1
        self.actor_model = actor_model.to(device)
        self.model = model
        self.target_model = copy.deepcopy(self.model)
        self.model.to(device)
        self.target_model.to(device)
        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.model.get_critic_params(), lr=self.critic_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.actor_lr)

    def predict(self, obs):
        self.model.eval()
        with torch.no_grad():
            obs = torch.tensor(obs, dtype=torch.float32)
            action, _, _ = self.actor_model.policy(obs)
        return action.cpu().numpy()

    def predict_e(self, obs):
        self.model.eval()
        with torch.no_grad():
            obs = torch.tensor(obs, dtype=torch.float32)
            action = self.actor_model.predict(obs)
        return action.cpu().numpy()

    def learn(self, obs, action, reward, next_obs, terminal, global_step):
        self.model.train()
        critic_loss = self._critic_learn(obs, action, reward, next_obs, terminal)
        actor_loss = self._actor_learn(obs)
        return actor_loss, critic_loss

    def _critic_learn(self, obs, action, reward, next_obs, terminal):
        with torch.no_grad():
            reward = reward.unsqueeze(1)
            terminal = terminal.unsqueeze(1)
            next_action, next_log_pro, _ = self.actor_model.policy(next_obs)
            next_entropy = -next_log_pro

            target_q1, target_q2 = self.target_model.value(next_obs, next_action)
            target_q = torch.min(target_q1, target_q2) + self.log_alpha.exp() * next_entropy
            terminal = torch.tensor(terminal, dtype=torch.float32)
            target_q = reward + (1.0 - terminal) * self.gamma * target_q
            target_q = target_q.detach()

        current_q1, current_q2 = self.model.value(obs, action.unsqueeze(1))
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss.item()

    def _actor_learn(self, obs):
        action, log_pro, pre_act = self.actor_model.policy(obs)
        entropy = - log_pro
        Q1, Q2 = self.model.value(obs, action)
        Q = torch.min(Q1, Q2)
        actor_loss = (-self.log_alpha.exp() * entropy - Q).mean()
        loss_alpha = self.log_alpha.exp() * (entropy - self.target_entropy).detach()
        loss_alpha = loss_alpha.mean()
        # 优化actor网络
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.alpha_optimizer.zero_grad()
        loss_alpha.backward()
        self.alpha_optimizer.step()
        return actor_loss.item()

    def sync_target(self, decay=None):
        if decay is None:
            decay = 1. - self.tau
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(decay * target_param.data + (1.0 - decay) * param.data)

    def save_model(self, path):
        torch.save({
            'actor_model_state_dict': self.actor_model.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.actor_model.load_state_dict(checkpoint['actor_model_state_dict'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        self.log_alpha = checkpoint['log_alpha']
        self.actor_model.eval()
        self.model.eval()
        self.target_model.eval()


class Agent(nn.Module):
    def __init__(self, algorithm, obs_dim, act_dim):
        super(Agent, self).__init__()
        assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.alg = algorithm

        self.global_step = 0
        self.update_target_steps = 2
        self.alg.sync_target(decay=0)
        self.a = 0

    def predict(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        action = self.alg.predict(obs)
        action = np.squeeze(action)
        return action

    def predict_e(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        action = self.alg.predict_e(obs)
        action = np.squeeze(action)
        return action

    def learn(self, obs, act, reward, next_obs, terminal):
        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target()
        self.global_step += 1
        obs = torch.tensor(obs, dtype=torch.float32).to(device)
        act = torch.tensor(act, dtype=torch.float32).to(device)
        reward = torch.tensor(reward, dtype=torch.float32).to(device)
        next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
        terminal = torch.tensor(terminal, dtype=torch.bool).to(device)

        actor_cost, critic_cost = self.alg.learn(obs, act, reward, next_obs,
                                                 terminal, self.global_step)
        return critic_cost, self.alg.log_alpha.exp()
