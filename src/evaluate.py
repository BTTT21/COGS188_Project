import time
import numpy as np
from game import Minesweeper

def evaluate_solver(solver_class, num_games=100):
    wins = 0
    total_time = 0
    exploration_rates = []
    
    for _ in range(num_games):
        game = Minesweeper()
        solver = solver_class(game)
        start_time = time.time()
        result = solver.train(1)  # 每次训练一个episode
        duration = time.time() - start_time
        
        if game.victory:
            wins += 1
        revealed = np.sum(game.revealed)
        exploration_rate = revealed / (game.rows*game.cols - game.total_mines)
        exploration_rates.append(exploration_rate)
        total_time += duration
    
    return {
        'win_rate': wins / num_games,
        'avg_time': total_time / num_games,
        'exploration_rate': np.mean(exploration_rates)
    }