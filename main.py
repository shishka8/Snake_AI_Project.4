import time
import os
import json
import matplotlib.pyplot as plt
from agent import Agent
from game import SnakeGameAI

HISTORY_FILE = './model/learning_history.json'

def generate_final_report(current_session_games, record, total_scores, total_mean_scores, session_marks):
    """Генерирует продвинутый сквозной график со всеми сессиями и разметками"""
    if len(total_scores) == 0: 
        return
        
    print("\n" + "="*50)
    print("        ОТЧЕТ ПО РЕЗУЛЬТАТАМ СЕССИИ ОБУЧЕНИЯ       ")
    print("="*50)
    print(f"Сыграно матчей за ЭТУ сессию: {current_session_games}")
    print(f"Всего сыграно матчей за ВСЁ время: {len(total_scores)}")
    print(f"Абсолютный рекорд: {max(total_scores)} очков")
    print("-"*50)
    
    # Создаем папку для отчетов, НАКОНЕЦТО ОНИ СОХРАНАЯЮТСЯ ИНЕ БУЖЕТ ОШБИКА ВЫЛЗЕАТЬ
    if not os.path.exists('./reports'):
        os.makedirs('./reports')
        
    plt.figure(figsize=(12, 6))
    plt.title('Сквозная динамика обучения ИИ (История всех сессий)')
    plt.xlabel('Общий номер игры (Все сессии)')
    plt.ylabel('Набранные очки')
    
    # Строим графики текущих результатов и сглаженного среднего значения
    plt.plot(total_scores, label='Очки в игре', color='#0000FF', alpha=0.4)
    plt.plot(total_mean_scores, label='Среднее значение', color='#FF0000', linewidth=2)
    
    # РАЗМЕТКА СЕССИЙ 
    # Рисуем вертикальные линии разделения сессий чтобы мы видели прогресс(нет)и историю запусков
    session_num = 1
    for mark in session_marks:
        plt.axvline(x=mark, color='gray', linestyle='--', alpha=0.7)
        plt.text(mark + 1, max(total_scores) * 0.9, f'Сессия {session_num}', 
                 fontsize=9, color='gray', rotation=90, verticalalignment='top')
        session_num += 1
        
    # Отдельно подписываем самую последнюю (текущую) сессию на графике
    if len(session_marks) > 0:
        last_mark = session_marks[-1]
        plt.text(last_mark + 1, max(total_scores) * 0.9, f'Сессия {session_num} (Текущая)', 
                 fontsize=9, color='darkgreen', rotation=90, verticalalignment='top')
    else:
        plt.text(1, max(total_scores) * 0.9, 'Сессия 1 (Текущая)', 
                 fontsize=9, color='darkgreen', rotation=90, verticalalignment='top')
    # -----------------------------
    
    plt.ylim(ymin=0)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend()
    
    # Сохраняем картинку с уникальным временным штампом чтобы не перезаписывать старые
    timestamp = int(time.time())
    report_img_path = f'./reports/global_report_{timestamp}.png'
    plt.savefig(report_img_path)
    print(f"[УСПЕХ] Глобальный график истории сохранен в: {report_img_path}")
    print("="*50 + "\n")
    plt.show()

def start_ai_project():
    print("==================================================")
    print("      ДОБРО ПОЖАЛОВАТЬ В ПРОЕКТ: ИИ ЗМЕЙКА       ")
    print("==================================================")
    print("Выберете режим запуска проекта:")
    print("  1. Режим тренировки ИИ (Обычное окно + сбор графиков)")
    print("  2. Играть против ИИ (Сплит-скрин, соревнование!)")
    print("==================================================")
    
    choice = input("Введите цифру режима (1 или 2): ").strip()
    
    if choice == '2':
        mode = 'vs'
        print("\nЗапуск режима дуэли! Управление вашей (ЗЕЛЕНОЙ) змейкой — СТРЕЛОЧКИ.")
    else:
        mode = 'train'
        print("\nЗапуск режима фоновой тренировки...")
        
    time.sleep(1.5)
    
    agent = Agent()
    
    # Подтягиваем всю старую инфу из json  чтобы графики не обнулялись
    total_scores, total_mean_scores, session_marks, global_games_counter = load_history()
    current_session_games = 0 
    
    # Если мы зашли в тренировку и у нас уже БЫЛА история, ставим отметку старта новой сессии
    if mode == 'train' and global_games_counter > 0:
        session_marks.append(global_games_counter)
    
    if mode == 'vs':
        agent.epsilon = 0 # В режиме дуэли полностью отключаем случайные шаги у ИИ, так винрейт мб вырастет у дуры этой
        
    game = SnakeGameAI(mode=mode)
    record = max(total_scores) if len(total_scores) > 0 else 0
    
    # Чтобы среднее продолжалось корректно, берем сумму всех старых очков
    running_total_score = sum(total_scores)

    try:
        while True:
            state_old = agent.get_state(game)
            
            if mode == 'vs':
                # Для режима игры вручную собираем решение на ходу через модель напрямую
                import torch
                state0 = torch.tensor(state_old, dtype=torch.float)
                prediction = agent.model(state0)
                move_idx = torch.argmax(prediction).item()
                final_move = [0, 0, 0]
                final_move[move_idx] = 1
            else:
                final_move = agent.get_action(state_old)

            reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game)

            if mode == 'train':
                # Закидываем этот шаг в краткосрочную и долгосрочную память агента
                agent.train_short_memory(state_old, final_move, reward, state_new, done)
                agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                game.reset()
                
                if mode == 'train':
                    agent.n_games += 1         
                    current_session_games += 1 
                    global_games_counter += 1  
                    
                    # Игра закончилась значит гоняем нейросеть по случайной пачке воспоминаний
                    agent.train_long_memory()

                    if score > record:
                        record = score
                        agent.model.save()

                    print(f'Сессия! Игра № {current_session_games} (Всего: {global_games_counter}) | Очки: {score} | Рекорд: {record}')
                    
                    # Математика сквозного скользящего среднего, чтобы линия на графике шла красиво, фу матиматика буэээ артем заучка
                    total_scores.append(score)
                    running_total_score += score
                    new_mean = running_total_score / global_games_counter
                    total_mean_scores.append(new_mean)
                    
                    # ЧТОБЫ ПРИ ВЫЛЕТЕ ОПЯТЬ ВСЕ НЕ СТЕРЛОСЬ Я НАГУГЛИЛ КАК МОЖНО ПОФИКСИТЬ
                    save_history(total_scores, total_mean_scores, session_marks, global_games_counter)
                else:
                    print(f'Матч окончен! ИИ набрал: {score} очков. Перезапуск дуэли...')
                    time.sleep(1)

    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if mode == 'train':
            # После закрытия проги сразу отчет вылезает и ниче не не надо нажимать ивсе такое
            generate_final_report(current_session_games, record, total_scores, total_mean_scores, session_marks)

# json функции или типа того

def load_history():
    """Загружает историю прошлых сессий из файла JSON"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return (data.get('scores', []), 
                        data.get('mean_scores', []), 
                        data.get('session_marks', []), 
                        data.get('total_games', 0))
        except Exception as e:
            print(f"Ошибка чтения истории: {e}. Начинаем с чистого листа.")
    return [], [], [], 0

def save_history(scores, mean_scores, session_marks, total_games):
    """Сохраняет историю в файл JSON"""
    if not os.path.exists('./model'):
        os.makedirs('./model')
    data = {
        'scores': scores,
        'mean_scores': mean_scores,
        'session_marks': session_marks,
        'total_games': total_games
    }
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    start_ai_project()