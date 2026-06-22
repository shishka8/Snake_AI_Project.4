import pygame
import random
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.SysFont('arial', 25)
Point = namedtuple('Point', 'x, y')

# Цвета
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
GREEN1 = (0, 200, 0)
GREEN2 = (0, 255, 100)
BLACK = (0, 0, 0)
GRAY = (50, 50, 50)

BLOCK_SIZE = 20
SPEED = 80

class SnakeGameAI:
    def __init__(self, mode='train'):
        self.mode = mode
        # Одиночное поле 640x480, для сплит-скрина расширяем в ширину до 1280
        self.w = 1280 if mode == 'vs' else 640
        self.h = 480
        self.field_w = 640 # Ширина одного игрового поля
        
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Змейка: ИИ против Человека' if mode == 'vs' else 'Обучение ИИ Змейки')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.frame_iteration = 0
        
        # --- ИНИЦИАЛИЗАЦИЯ ИИ (Она всегда на правом поле, если сплит-скрин)
        x_offset = self.field_w if self.mode == 'vs' else 0
        self.direction = 'RIGHT'
        self.head = Point(x_offset + self.field_w // 2, self.h // 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)
        ]
        self.score = 0
        self.food = None
        self._place_food()

        # --- ИНИЦИАЛИЗАЦИЯ ИГРОКА (Левое поле, только для режима 'vs')
        if self.mode == 'vs':
            self.p_direction = 'RIGHT'
            self.p_head = Point(self.field_w // 4, self.h // 2)
            self.p_snake = [
                self.p_head,
                Point(self.p_head.x - BLOCK_SIZE, self.p_head.y),
                Point(self.p_head.x - (2 * BLOCK_SIZE), self.p_head.y)
            ]
            self.p_score = 0
            self.p_food = None
            self._place_p_food()
            self.p_game_over = False

    def _place_food(self):
        x_offset = self.field_w if self.mode == 'vs' else 0
        x = random.randint(0, (self.field_w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE + x_offset
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def _place_p_food(self):
        x = random.randint(0, (self.field_w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.p_food = Point(x, y)
        if self.p_food in self.p_snake:
            self._place_p_food()

    def play_step(self, action):
        self.frame_iteration += 1
        
        # Считываем управление игрока (стрелочки)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                import sys
                sys.exit()
            if self.mode == 'vs' and event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and self.p_direction != 'RIGHT': self.p_direction = 'LEFT'
                elif event.key == pygame.K_RIGHT and self.p_direction != 'LEFT': self.p_direction = 'RIGHT'
                elif event.key == pygame.K_UP and self.p_direction != 'DOWN': self.p_direction = 'UP'
                elif event.key == pygame.K_DOWN and self.p_direction != 'UP': self.p_direction = 'DOWN'

        # 1. ДВИЖЕНИЕ ИГРОКА (если режим VS и он еще жив)
        if self.mode == 'vs' and not self.p_game_over:
            self._move_player()
            self.p_snake.insert(0, self.p_head)
            if self._is_player_collision():
                self.p_game_over = True
            elif self.p_head == self.p_food:
                self.p_score += 1
                self._place_p_food()
            else:
                self.p_snake.pop()

        # 2. ДВИЖЕНИЕ ИИ
        self._move_ai(action)
        self.snake.insert(0, self.head)
        
        reward = 0
        ai_game_over = False
        
        # Проверка смерти ИИ (или зацикливания)
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            ai_game_over = True
            reward = -10
            # В режиме VS игра перезапускается, только если ОБА проиграли или ИИ умер
            return reward, True, self.score

        # Компас наград для ИИ
        old_dist = np.sqrt((self.snake[1].x - self.food.x)**2 + (self.snake[1].y - self.food.y)**2)
        new_dist = np.sqrt((self.head.x - self.food.x)**2 + (self.head.y - self.food.y)**2)

        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
            self.frame_iteration = 0
        else:
            self.snake.pop()
            if new_dist < old_dist: reward = 0.1
            else: reward = -0.15

        # 3. ОТРИСОВКА ЭКРАНА
        self._update_ui()
        self.clock.tick(SPEED)
        
        # В режиме VS возвращаем game_over, если ИИ умер или Игрок умер
        game_finished = ai_game_over or (self.mode == 'vs' and self.p_game_over)
        return reward, game_finished, self.score

    def is_collision(self, pt=None):
        if pt is None: pt = self.head
        x_offset = self.field_w if self.mode == 'vs' else 0
        if pt.x > (x_offset + self.field_w - BLOCK_SIZE) or pt.x < x_offset or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def _is_player_collision(self):
        if self.p_head.x > self.field_w - BLOCK_SIZE or self.p_head.x < 0 or self.p_head.y > self.h - BLOCK_SIZE or self.p_head.y < 0:
            return True
        if self.p_head in self.p_snake[1:]:
            return True
        return False

    def _move_player(self):
        x, y = self.p_head.x, self.p_head.y
        if self.p_direction == 'RIGHT': x += BLOCK_SIZE
        elif self.p_direction == 'LEFT': x -= BLOCK_SIZE
        elif self.p_direction == 'DOWN': y += BLOCK_SIZE
        elif self.p_direction == 'UP': y -= BLOCK_SIZE
        self.p_head = Point(x, y)

    def _move_ai(self, action):
        clock_wise = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        idx = clock_wise.index(self.direction)
        if np.array_equal(action, [1, 0, 0]): new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]): new_dir = clock_wise[(idx + 1) % 4]
        else: new_dir = clock_wise[(idx - 1) % 4]
        self.direction = new_dir

        x, y = self.head.x, self.head.y
        if self.direction == 'RIGHT': x += BLOCK_SIZE
        elif self.direction == 'LEFT': x -= BLOCK_SIZE
        elif self.direction == 'DOWN': y += BLOCK_SIZE
        elif self.direction == 'UP': y -= BLOCK_SIZE
        self.head = Point(x, y)

    def _update_ui(self):
        self.display.fill(BLACK)
        
        # --- РИСУЕМ ПОЛЕ ИИ ---
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        # Текст очков ИИ
        x_txt_offset = self.field_w if self.mode == 'vs' else 0
        text = font.render(f"ИИ Очки: {self.score}", True, WHITE)
        self.display.blit(text, [x_txt_offset + 10, 10])

        # --- РИСУЕМ ПОЛЕ ИГРОКА (Только в режиме 'vs') ---
        if self.mode == 'vs':
            # Линия разделения экранов
            pygame.draw.line(self.display, GRAY, (self.field_w, 0), (self.field_w, self.h), 5)
            
            # Рисуем игрока
            for pt in self.p_snake:
                pygame.draw.rect(self.display, GREEN1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, GREEN2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
            pygame.draw.rect(self.display, RED, pygame.Rect(self.p_food.x, self.p_food.y, BLOCK_SIZE, BLOCK_SIZE))
            
            # Текст очков Игрока
            p_text = font.render(f"Вы: {self.p_score}", True, WHITE)
            self.display.blit(p_text, [10, 10])
            
            if self.p_game_over:
                lost_text = font.render("ВЫ ПРОИГРАЛИ! Ожидание ИИ...", True, RED)
                self.display.blit(lost_text, [self.field_w // 4 - 50, self.h // 2])

        pygame.display.flip()