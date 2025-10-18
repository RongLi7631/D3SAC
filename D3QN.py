import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import numpy as np

device = torch.device("cpu")#torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Model, self).__init__()
        hid_size = 128
        self.fc1_adv = nn.Linear(in_features=obs_dim, out_features=hid_size)
        self.fc2_adv = nn.Linear(in_features=hid_size, out_features=hid_size)
        self.fc3_adv = nn.Linear(in_features=hid_size, out_features=act_dim)

        self.fc1_val = nn.Linear(in_features=obs_dim, out_features=hid_size)
        self.fc2_val = nn.Linear(in_features=hid_size, out_features=hid_size)
        self.fc3_val = nn.Linear(in_features=hid_size, out_features=1)

    def forward(self, obs):
        adv = F.relu(self.fc1_adv(obs))
        adv = F.relu(self.fc2_adv(adv))
        As = self.fc3_adv(adv)
        val = F.relu(self.fc1_val(obs))
        val = F.relu(self.fc2_val(val))
        V = self.fc3_val(val)
        Q = As + (V - As.mean(dim=1, keepdim=True))
        return Q


class DDQN(nn.Module):
    def __init__(self, model, act_dim, gamma, lr):
        super(DDQN, self).__init__()

        self.model = model
        self.target_model = copy.deepcopy(model)
        self.model.to(device)
        self.target_model.to(device)
        self.act_dim = act_dim
        self.gamma = gamma
        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def predict(self, obs):
        self.model.eval()
        return self.model(obs)

    def learn(self, obs, action, reward, next_obs, terminal, learning_rate=None):
        self.model.train()
        if learning_rate is None:
            learning_rate = self.lr
        pred_value = self.model(obs)
        action_onehot = F.one_hot(action, num_classes=self.act_dim).float()
        pred_action_value = (action_onehot * pred_value).sum(dim=1)
        next_action_value = self.model(next_obs)
        greedy_action = next_action_value.argmax(dim=1)
        next_pred_value = self.target_model(next_obs)
        max_v = next_pred_value.gather(1, greedy_action.unsqueeze(1).long())
        max_v = max_v.squeeze(1)
        target = reward + (1.0 - terminal.float()) * self.gamma * max_v.detach()
        loss = F.mse_loss(pred_action_value, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def sync_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.eval()
        self.target_model.eval()


class Agent:
    def __init__(self, algorithm, obs_dim, act_dim, e_greed=0.1, e_greed_decrement=0):
        assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.alg = algorithm
        self.global_step = 0
        self.update_target_steps = 2
        self.e_greed = e_greed
        self.e_greed_decrement = e_greed_decrement

    def sample(self, obs):
        sample = np.random.rand()
        if sample < self.e_greed:
            act = np.random.randint(self.act_dim)
        else:
            act = self.predict(obs)
        self.e_greed = max(0.01, self.e_greed - self.e_greed_decrement)
        return act

    def predict(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_Q = self.alg.predict(obs)
        act = torch.argmax(pred_Q, dim=-1).item()
        return act

    def learn(self, obs, act, reward, next_obs, terminal):
        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target()
        self.global_step += 1
        obs = torch.tensor(obs, dtype=torch.float32).to(device)
        act = torch.tensor(act, dtype=torch.long).to(device)
        reward = torch.tensor(reward, dtype=torch.float32).to(device)
        next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
        terminal = torch.tensor(terminal, dtype=torch.bool).to(device)

        loss = self.alg.learn(obs, act, reward, next_obs, terminal)
        return loss.item()
