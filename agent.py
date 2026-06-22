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
        self.epsilon = 0 # Фактор рандома (исследования мира)
        self.gamma = 0.9 # Коэффициент дисконтирования для формулы Беллмана (точняк Беллман звали его, из инета прост формулу взял, артем не сердись)
        self.memory = deque(maxlen=MAX_MEMORY) #  память агента
        
        # Тут я крч вернул 11 входов вместо 16 или 24 круговых, так эта тупая тварь не должна тупить и кружится,,, ПОЧИНИЛ ТЕП%ЕРЬ ОНА ОБУЧАЕТСЯ УРААА
        self.model = Linear_QNet(11, 256, 3)  
        
        # Автоматическая загрузка сохраненных весов модели
        model_path = './model/model.pth'
        if os.path.exists(model_path):
            print("=========================================")
            print(" ЗАГРУЖАЮ СТАБИЛЬНОЕ БЛИЖНЕЕ ЗРЕНИЕ...")  #РАБОТАЕТ УРААА
            print("=========================================")
            try:
                self.model.load_state_dict(torch.load(model_path))
                self.model.eval()
                self.n_games = 50 
            except:
                print("Архитектура изменилась, начинаем чистое обучение.")
            
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_action(self, state):
        # Крч, чем больше игр сыграно, тем меньше рандомных шагов и тем больше змейка доверяет своей нейросети, мразь тупая необучаемаяя
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        
        if random.randint(0, 200) < self.epsilon:
            # чтобы изучать она рандомно поповрачивает голову
            move_idx = random.randint(0, 2)
            final_move[move_idx] = 1
        else:
            # выбираем лучшее действие
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move_idx = torch.argmax(prediction).item()
            final_move[move_idx] = 1

        return final_move

    def get_state(self, game):
        head = game.head
        # Вычисляем соседние точки вокруг головы для проверки столкновений
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        
        # Текущее направление движения змейки
        dir_l = game.direction == 'LEFT'
        dir_r = game.direction == 'RIGHT'
        dir_u = game.direction == 'UP'
        dir_d = game.direction == 'DOWN'

        # Формируем итоговый вектор из 11 бинарных признаков (наше зрение)
        state = [
            # Опасность прямо 
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Опасность справа 
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Опасность слева 
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Направление взгляда 
            dir_l, dir_r, dir_u, dir_d,
            # блок цели
            # Положение яблока относительно головы змейки
            game.food.x < game.head.x,  # Еда левее
            game.food.x > game.head.x,  # Еда правее
            game.food.y < game.head.y,  # Еда выше
            game.food.y > game.head.y   # Еда ниже
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        # Запись текущего шага в буфер памяти для последующего обучения-----Артем я добавил буфер памяти как и просил а то это говно после каждой сессии сбрасывало прогресс и все зря((
        self.memory.append((state, action, reward, next_state, done))

    def train_short_memory(self, state, action, reward, next_state, done):
        # Быстрое обучение на основе только что сделанного шага
        self.trainer.train_step(state, action, reward, next_state, done)

    def train_long_memory(self):
        # Обучение на большой выборке рандомных шагов из памяти 
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)