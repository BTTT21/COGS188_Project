import time
import numpy as np
from game import Minesweeper

def evaluate_solver(solver_class, num_games=100):
    wins = 0
    total_time = 0
    exploration_rates = []
    best_episode = None
    best_score = -float('inf')  # 定义一个初始最小得分

    for episode in range(num_games):
        game = Minesweeper()
        solver = solver_class(game)
        start_time = time.time()
        solver.train(1)  # 每个 episode 训练一次
        duration = time.time() - start_time
        
        # 计算当前 episode 的探索率：已揭示单元格 / (总格数 - 雷数)
        revealed = np.sum(game.revealed)
        exploration_rate = revealed / (game.rows * game.cols - game.total_mines)
        exploration_rates.append(exploration_rate)
        total_time += duration
        
        # 定义一个简单的得分标准：获胜 episode 得分为探索率，否则得分为0（也可以自行设计其他得分规则）
        if game.victory:
            wins += 1
            score = exploration_rate
        else:
            score = 0
        
        # 如果当前 episode 得分更高，则记录下来
        if score > best_score:
            best_score = score
            best_episode = {
                'episode': episode,
                'game': game,
                'solver': solver,
                'duration': duration,
                'exploration_rate': exploration_rate,
                'victory': game.victory
            }
    
    evaluation_metrics = {
        'win_rate': wins / num_games,
        'avg_time': total_time / num_games,
        'avg_exploration_rate': np.mean(exploration_rates)
    }
    return evaluation_metrics, best_episode