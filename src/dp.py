import numpy as np
import time
from collections import defaultdict
from solver import BaseSolver

class DPSolver(BaseSolver):
    def __init__(self, game, gamma=0.9, max_iter=100):
        super().__init__(game)
        self.gamma = gamma
        self.max_iter = max_iter
        self.value_table = np.zeros((game.rows, game.cols))
        self.reward_safe = 1
        self.reward_mine = -100
        
    def value_iteration(self):
        for _ in range(self.max_iter):
            new_table = self.value_table.copy()
            for r in range(self.game.rows):
                for c in range(self.game.cols):
                    # 只更新未揭示且未标记的格子
                    if self.game.revealed[r, c] or self.game.flags[r, c]:
                        continue
                    # 计算相邻单元格的预估价值
                    neighbors = self.game._get_neighbors(r, c)
                    neighbor_values = []
                    for nr, nc in neighbors:
                        if self.game.revealed[nr, nc]:
                            val = self.value_table[nr, nc]
                        else:
                            # 未揭示单元格先验设为安全奖励
                            val = self.reward_safe
                        neighbor_values.append(val)
                    # 采用50%概率安全的假设更新当前格子价值
                    avg_value = 0.5 * (self.reward_safe + self.gamma * np.mean(neighbor_values))
                    new_table[r, c] = avg_value
            # 当新旧价值足够接近时提前退出
            if np.allclose(new_table, self.value_table, atol=1e-4):
                break
            self.value_table = new_table
        
    def select_action(self):
        candidates = np.argwhere(~self.game.revealed & ~self.game.flags)
        if len(candidates) == 0:
            return None
        # 从候选格子中选取价值最高的
        values = [self.value_table[r, c] for (r, c) in candidates]
        best_idx = np.argmax(values)
        return (*candidates[best_idx], 'reveal')
        
    def train(self, episodes):
        for _ in range(episodes):
            self.game.reset()
            while not self.game.game_over:
                self.value_iteration()
                action = self.select_action()
                if action is None:
                    break
                r, c, _ = action
                self.game.reveal(r, c)