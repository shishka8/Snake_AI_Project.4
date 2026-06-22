import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Тут я крч поменял немного мозги змейке, чтобы больше не на 8 направлений во все стороны смотрела, а попроще, как до этого было (на 11 входов)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Активация ReLU для добавления нелинейности в модель
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        # Проверка наличия директории перед сохранением весов
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        # Сохраняем состояние модели чтобы не терять прогресс при перезапуске
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        # оптимизатор контролирует шаг или типа того
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss() # Среднеквадратичная ошибка для оценки функции потерь

    def train_step(self, state, action, reward, next_state, done):
        # Преобразование входных данных в тензоры PyTorch
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)

        if len(state.shape) == 1:
            # Корректировка размерности, если на вход пришел один шаг, а не пакет данных (batch)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # Получаем текущие предсказания от модели
        pred = self.model(state)
        target = pred.clone()

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                # Короче считает ценность по какой то там формуле из инета, сложная 0_0
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # Стандартный процесс обратного распространения ошибки и обновления весов сети
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()