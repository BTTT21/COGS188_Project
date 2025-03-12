import random
import pygame
from minesweeper_adapted_for_MC import Game, LEFT_CLICK
import matplotlib.pyplot as plt
import numpy as np


class OffPolicyMonteCarloSolver:

    def __init__(self, game, episodes=1000, gamma=1.0):
        self.game = game
        self.episodes = episodes
        self.gamma = gamma
        self.Q = {}  # (state, action) -> value
        self.returns_sum = {}  # (state, action) -> total return
        self.returns_count = {}  # (state, action) -> count
        self.policy = {}  # state -> action
        self.train_results = []  # 用于记录每轮的胜负

    def observe_state(self):
        return tuple(tuple(1 if cell.is_visible else 0 for cell in row) for row in self.game.grid)


    def behavior_policy(self, state, first_click_done, epsilon=0.3):
        # --- 随机探索：以 ε 概率进行随机探索 ---
        if random.random() < epsilon:
            unknown = self.get_unknown_cells()
            if unknown:
                print("Exploration action!")
                return random.choice(unknown)

        # --- 第一次随机点击一个位置 ---
        if not first_click_done:
            unknown = self.get_unknown_cells()
            return random.choice(unknown) if unknown else None

        # --- 扫描所有可见数字，寻找安全推测格 ---
        for r in range(self.game.squares_y):
            for c in range(self.game.squares_x):
                cell = self.game.grid[r][c]
                if cell.is_visible and cell.bomb_count > 0:
                    # 查找未点的邻居
                    neighbors = []
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < self.game.squares_y and 0 <= nc < self.game.squares_x:
                                neighbor = self.game.grid[nr][nc]
                                if not neighbor.is_visible and not neighbor.has_flag:
                                    neighbors.append((nr, nc))
                    # 如果未知邻居只有一个，优先尝试
                    if len(neighbors) == 1:
                        return neighbors[0]

        # --- 无法推理则随机 ---
        unknown = self.get_unknown_cells()
        return random.choice(unknown) if unknown else None



    def get_unknown_cells(self):
        return [(r, c) for r in range(self.game.squares_y) for c in range(self.game.squares_x)
                if not self.game.grid[r][c].is_visible and not self.game.grid[r][c].has_flag]

    def click_cell(self, row, col):
        print(f"Clicking on cell: ({row}, {col})")  # 打印点击的坐标
        self.game.click_handle(row, col, LEFT_CLICK)
        self.game.check_victory()

    def generate_episode(self, max_steps=100):  # 强制步数上限
        episode = []
        self.game.reset_game(keep_bombs=False)
        visited = set()
        step = 0
        first_click_done = False  # 初始设为 False，标识是否首次点击

        while not self.game.game_lost and not self.game.game_won:
            if step >= max_steps:
                print(f"Episode reached max steps {max_steps}, breaking!")
                break

            state = self.observe_state()
            action = self.behavior_policy(state, first_click_done)  # 传入 first_click_done
            first_click_done = True  # 第一次点击后标记为 True

            if action is None or (state, action) in visited:
                print(f"No valid action or revisited (state, action), breaking!")
                break

            self.click_cell(*action)
            reward = 1 if self.game.game_won else (-1 if self.game.game_lost else 0)
            episode.append((state, action, reward))
            visited.add((state, action))
            step += 1

        return episode



    def update_policy_from_episode(self, episode, tau=1.0, min_count=5):
        G = 0
        for state, action, reward in reversed(episode):
            G = self.gamma * G + reward
            sa = (state, action)
        
            # 衰减式更新 Q
            self.returns_count[sa] = self.returns_count.get(sa, 0) + 1
            alpha = 1 / self.returns_count[sa]
            self.Q[sa] = self.Q.get(sa, 0) + alpha * (G - self.Q.get(sa, 0))
        
            # 更新策略，基于 softmax 但仅在有足够样本的情况下
            state_actions = [a for (s, a) in self.Q.keys() if s == state and self.returns_count[(s, a)] >= min_count]
            if state_actions:
                q_values = np.array([self.Q[(state, a)] for a in state_actions])
                exp_q = np.exp(q_values / tau)
                probs = exp_q / np.sum(exp_q)
                # 保存在 policy 里完整分布而非单一 action
                self.policy[state] = (state_actions, probs)


    def run_training(self):
        for ep in range(1, self.episodes + 1):
            print(f"--- Episode {ep} ---")
            episode = self.generate_episode()
            self.update_policy_from_episode(episode)
            final_reward = episode[-1][2] if episode else 0
            result = 'Win' if final_reward == 1 else 'Loss'
            print(f"{result} | Steps: {len(episode)}")
            self.train_results.append(1 if final_reward == 1 else 0)




    def play_with_policy(self):
        self.game.reset_game(keep_bombs=False)
        while not self.game.game_lost and not self.game.game_won:
            state = self.observe_state()
            action = self.policy.get(state)
            if action is None:
                break
            self.click_cell(*action)
        return self.game.game_won
    

    def plot_training_results(self):
        win_rate = [sum(self.train_results[:i+1]) / (i+1) for i in range(len(self.train_results))]
        plt.figure(figsize=(10, 5))
    
        # 只绘制从第 100 个 episode 开始的数据
        episodes = list(range(1, len(self.train_results) + 1))  # [1, 2, ..., N]
        plt.plot(episodes[99:], win_rate[99:], marker='o', markersize=2, linewidth=1, label='Win Rate')
    
        plt.xlabel('Episodes')
        plt.ylabel('Win Rate')
        plt.title('Training Win Rate over Episodes (from Episode 100)')
        plt.legend()
        plt.show()



# --- 主程序 ---
if __name__ == "__main__":
    game = Game(use_display=False, num_bombs=5, fixed_seed=None)
    solver = OffPolicyMonteCarloSolver(game, episodes=500)  # 自定义训练轮数
    solver.run_training()  # 训练
    solver.plot_training_results()  # 绘制 win-rate 曲线

    # 测试学到的策略
    wins = 0
    trials = 100
    for _ in range(trials):
        if solver.play_with_policy():
            wins += 1
    print(f"Win rate over {trials} trials: {wins / trials:.2%}")



    