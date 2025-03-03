import numpy as np
import time
from collections import defaultdict
from solver import BaseSolver

class MCSolver(BaseSolver):
    def __init__(self, game, simulations=500, epsilon=0.1):
        super().__init__(game)
        self.simulations = simulations
        self.epsilon = epsilon
        self.Q = defaultdict(float)  # 动作价值
        self.N = defaultdict(int)    # 访问次数
        
    def run_simulation(self):
        state = self._get_state_hash()
        original_state = self.game.get_observation()
        total_rewards = defaultdict(float)
        
        for _ in range(self.simulations):
            # 重置游戏状态为初始状态
            self.game.__dict__.update(original_state)
            steps = []
            while not self.game.game_over:
                # ε-greedy动作选择
                if np.random.random() < self.epsilon:
                    action = self._random_action()
                else:
                    action = self._best_action()
                
                # 如果没有可选动作，跳出循环
                if action is None:
                    break
                
                r, c, _ = action
                prev_state = self._get_state_hash()
                reward = 1 if self.game.reveal(r, c) else -100
                steps.append((prev_state, action, reward))
            
            # 如果当前模拟没有任何步，说明没有可行动作，直接结束模拟
            if not steps:
                break

            # 反向传播回报
            G = 0
            for (s, a, r) in reversed(steps):
                G = r + 0.9 * G  # 折扣因子
                self.Q[(s, a)] += G
                self.N[(s, a)] += 1
        
    def _get_state_hash(self):
        return hash(self.game.revealed.tobytes())
        
    def _random_action(self):
        candidates = np.argwhere(~self.game.revealed & ~self.game.flags)
        if len(candidates) == 0:
            return None
        r, c = candidates[np.random.choice(len(candidates))]
        return (r, c, 'reveal')
        
    def _best_action(self):
        candidates = np.argwhere(~self.game.revealed & ~self.game.flags)
        if candidates.size == 0:
            return None
        state = self._get_state_hash()
        best_value = -float('inf')
        best_action = None
        for (r, c) in candidates:
            action = (r, c, 'reveal')
            key = (state, action)
            value = self.Q[key] / self.N[key] if self.N[key] > 0 else 0
            if value > best_value:
                best_value = value
                best_action = action
        # 如果没有选出最优动作，则退回到随机动作
        return best_action or self._random_action()
        
    def train(self, episodes):
        for _ in range(episodes):
            self.run_simulation()