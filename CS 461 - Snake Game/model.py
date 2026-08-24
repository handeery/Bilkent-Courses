import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
class DuelingLinear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DuelingLinear_QNet, self).__init__()

        self.common_stream = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, output_size)
        )

        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        x = self.common_stream(x)

        advantage = self.advantage_stream(x)
        value = self.value_stream(x)

        # Combine value and advantage to get Q values
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))

        return q_values
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
class DoubleQTrainer:
    def __init__(self, model, target_model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.target_model = target_model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # 1: predicted Q values with current state
        pred = self.model(state)

        # 2: Use Q-network to select actions
        pred_actions = torch.argmax(pred, dim=1)

        # 3: Get Q values from the target model for the selected actions
        target_pred = self.target_model(next_state).detach()
        selected_q_values = target_pred[range(len(pred_actions)), pred_actions]

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                # 4: Double DQN update rule
                Q_new = reward[idx] + self.gamma * selected_q_values[idx]

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # 5: Update the Q-network
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
class DuelingQTrainer:
    def __init__(self, model, target_model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.target_model = target_model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    def train_step(self, states, actions, rewards, next_states, dones):
        states = torch.tensor(states, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.bool)

        if len(states.shape) == 1:
            states = torch.unsqueeze(states, 0)
            next_states = torch.unsqueeze(next_states, 0)
            actions = torch.unsqueeze(actions, 0)
            rewards = torch.unsqueeze(rewards, 0)
            dones = torch.unsqueeze(dones, 0)

        # 1: predicted Q values with current state
        pred = self.model(states)

        # 2: Use Q-network to select actions
        pred_actions = torch.argmax(pred, dim=1)

        # 3: Get Q values from the target model for the selected actions
        target_pred = self.target_model(next_states).detach()
        selected_q_values = target_pred[range(len(pred_actions)), pred_actions]

        # 4: Separate the value and advantage streams from the predicted Q values
        pred_value = pred.mean(dim=1, keepdim=True)
        pred_advantage = pred - pred_value

        # 5: Get the Q values for the next state from the value and advantage streams
        target_value = target_pred.mean(dim=1, keepdim=True)
        target_advantage = target_pred - target_value

        # 6: Double DQN update rule
        Q_new = rewards + self.gamma * target_value.squeeze() + (~dones) * self.gamma * target_advantage[range(len(pred_actions)), pred_actions]

        # 7: Update the Q values for the selected actions
        pred[range(len(pred_actions)), pred_actions] = Q_new

        # 8: Update the Q-network
        self.optimizer.zero_grad()
        loss = F.mse_loss(pred, self.model(states))
        loss.backward()
        self.optimizer.step()