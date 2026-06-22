import torch
import random
import numpy as np
from collections import deque
import os
from game import SnakeGameAI, Point, BLOCK_SIZE
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        
        # Возвращаем 11 входов
        self.model = Linear_QNet(11, 256, 3)  
        
        # Автозагрузка весов
        model_path = './model/model.pth'
        if os.path.exists(model_path):
            print("=========================================")
            print(" ЗАГРУЖАЮ СТАБИЛЬНОЕ БЛИЖНЕЕ ЗРЕНИЕ...")
            print("=========================================")
            try:
                self.model.load_state_dict(torch.load(model_path))
                self.model.eval()
                self.n_games = 50 
            except:
                print("Архитектура изменилась, начинаем чистое обучение.")
            
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.head
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        
        dir_l = game.direction == 'LEFT'
        dir_r = game.direction == 'RIGHT'
        dir_u = game.direction == 'UP'
        dir_d = game.direction == 'DOWN'

        # Тот самый оригинальный вектор из 11 бинарных признаков
        state = [
            # Опасность прямо по ходу движения
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Опасность справа по ходу движения
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Опасность слева по ходу движения
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Направление взгляда (вектор движения)
            dir_l, dir_r, dir_u, dir_d,
            
            # Где яблоко относительно головы
            game.food.x < game.head.x,  # Еда левее
            game.food.x > game.head.x,  # Еда правее
            game.food.y < game.head.y,  # Еда выше
            game.food.y > game.head.y   # Еда ниже
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        
        if random.randint(0, 200) < self.epsilon:
            move_idx = random.randint(0, 2)
            final_move[move_idx] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move_idx = torch.argmax(prediction).item()
            final_move[move_idx] = 1

        return final_move