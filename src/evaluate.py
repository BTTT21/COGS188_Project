import time
import numpy as np
from game import Minesweeper

def evaluate_solver(solver_class, num_games=100):
    wins = 0
    total_time = 0
    exploration_rates = []
    best_episode = None
    best_score = -float('inf')  # 初始最小得分

    for episode in range(num_games):
        game = Minesweeper()
        solver = solver_class(game)
        start_time = time.time()
        solver.train(1)  # 每个 episode 训练一次
        duration = time.time() - start_time
        
        # 只计算安全格子的揭示数量（排除地雷）
        safe_revealed = np.sum((game.revealed) & (game.board != -1))
        safe_total = game.rows * game.cols - game.total_mines
        exploration_rate = safe_revealed / safe_total
        exploration_rates.append(exploration_rate)
        total_time += duration
        
        # 简单得分：获胜时得分为探索率，否则为 0
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