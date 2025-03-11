import matplotlib.pyplot as plt
import random
import pygame
from minesweeper_adapted_for_MC import Game, LEFT_CLICK


class MonteCarloSolver:

    def __init__(self, game, simulations=3, seed=42, verbose = False):
        self.game = game
        self.simulations = simulations
        self.seed = seed
        self.safe_memory = set()
        self.last_click = None
        self.steps = 0
        self.verbose = verbose

    def run_episode(self, max_steps=300):
        self.steps = 0
        self.game.reset_game(keep_bombs=True)
        self.last_click = None

        # 先点记忆安全格子
        self.click_known_safe_cells()

        # 初次随机点击
        unknown = self.get_unknown_cells()
        if unknown and not self.safe_memory:
            first_click = random.choice(unknown)
            self.click_cell(*first_click)

        # 游戏循环
        while not self.game.game_lost and not self.game.game_won and self.steps < max_steps:

            if not self.make_best_move():
                # 初次随机点击
                unknown = self.get_unknown_cells()
                if unknown and not self.safe_memory:
                    next_click = random.choice(unknown)
                    self.click_cell(*next_click)
                    print("随机点击")


        return self.game.game_won, self.steps
    


    # 获取未知格子（按是否在记忆中排序）
    def get_unknown_cells(self):
        return [(r, c) for r in range(self.game.squares_y) for c in range(self.game.squares_x)
            if not self.game.grid[r][c].is_visible and not self.game.grid[r][c].has_flag and (r, c) not in self.safe_memory]


    # 点击封装，统一计步、更新点击
    def click_cell(self, row, col):
        self.game.click_handle(row, col, LEFT_CLICK)  # 执行点击
        self.steps += 1
        self.last_click = (row, col)
        self.game.check_victory()

        # 如果点开的不是雷，加入记忆
        if not self.game.grid[row][col].has_bomb and (row, col) not in self.safe_memory:
            self.safe_memory.add((row, col))
    


    # 记忆安全格子点击，返回是否有点击
    def click_known_safe_cells(self):
        moved = False
        for row, col in list(self.safe_memory):
            if not self.game.grid[row][col].is_visible:
                print(f"记忆点击 ({row}, {col})")
                self.click_cell(row, col)
                moved = True
        return moved


    # Monte Carlo 分析
    def monte_carlo_analysis(self):
        unknown = self.get_unknown_cells()

        if not unknown:
            return None
        
        constraints = [(r, c) for r in range(self.game.squares_y) for c in range(self.game.squares_x)
                       if self.game.grid[r][c].is_visible and self.game.grid[r][c].bomb_count > 0]
        num_mines_left = self.game.num_bombs
        mine_prob = {cell: 0 for cell in unknown}
        valid_samples = 0

        for _ in range(self.simulations * 10):
            mines = set(random.sample(unknown, min(num_mines_left, len(unknown))))
            if self.is_valid_minefield(mines, constraints):
                valid_samples += 1
                for cell in mines:
                    mine_prob[cell] += 1
            if valid_samples >= self.simulations:
                break

        if valid_samples == 0:
            return None  # 没有有效样本返回 None

        # 返回当前踩雷概率最低的格子
        best_move = min(mine_prob, key=lambda x: mine_prob[x])
        if self.verbose:
            print(f"MC分析选择: {best_move}，估算雷率: {mine_prob[best_move] / valid_samples:.2%}")
        return best_move
    


    # 验证合理雷区
    def is_valid_minefield(self, minefield, constraints):
        for r, c in constraints:
            expected = self.game.grid[r][c].bomb_count
            neighbors = [(r + dx, c + dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1]
                         if (dx != 0 or dy != 0) and 0 <= r + dx < self.game.squares_y and 0 <= c + dy < self.game.squares_x]
            count = sum(1 for n in neighbors if n in minefield)
            if count != expected:
                return False
        return True

    # 决策流程
    def make_best_move(self):

        # 1. MC 推测
        best_move = self.monte_carlo_analysis()
        if best_move:
            print(f"MC点击 {best_move}")
            self.click_cell(*best_move)
            return True

        # 2. 无奈随机点击
        unknown_cells = self.get_unknown_cells()
        if unknown_cells:
            fallback = random.choice(unknown_cells)
            print(f"无奈点击: {fallback}")
            self.click_cell(*fallback)
            return True

        return False


# --- 训练与绘图函数 ---
def run_training(solver, num_episodes=90):
    results, steps = [], []
    for ep in range(1, num_episodes + 1):
        print(f"\n--- Episode {ep} ---")
        win, step = solver.run_episode()
        results.append(1 if win else 0)
        steps.append(step)
        print(f"{'Win' if win else 'Loss'} in {step} steps")
    return results, steps


def plot_training_results(results, steps):
    win_rate = [sum(results[:i + 1]) / (i + 1) for i in range(len(results))]
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(win_rate, marker='o', label='Win Rate')
    plt.xlabel("Episodes")
    plt.ylabel("Win Rate")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(steps, color='orange', marker='o', label='Steps')
    plt.xlabel("Episodes")
    plt.ylabel("Steps per Episode")
    plt.legend()
    plt.show()


# --- 主程序 ---
if __name__ == "__main__":
    game = Game(use_display=False)
    solver = MonteCarloSolver(game, simulations=3, seed=31)
    results, steps = run_training(solver, num_episodes=90)
    plot_training_results(results, steps)

