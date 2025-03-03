import pygame
import sys
import numpy as np
import random
from random import sample
from minesweeper_cited import Game, Menu, LEFT_CLICK, WIDTH, MARGIN, MENU_SIZE, HEIGHT, clock


# 创建游戏实例
game = Game()
menu = Menu()

# Monte Carlo AI Solver
class MonteCarloSolver:
    def __init__(self, game, simulations=1000):
        self.game = game
        self.simulations = simulations  # 采样次数

    def get_unknown_cells(self):
        """ 获取所有未翻开的格子 """
        return [(row, col) for row in range(self.game.squares_y) 
                for col in range(self.game.squares_x) 
                if not self.game.grid[row][col].is_visible and not self.game.grid[row][col].has_flag]

    def get_constraint_cells(self):
        """ 获取所有已翻开的、带数字的格子 """
        return [(row, col) for row in range(self.game.squares_y) 
                for col in range(self.game.squares_x) 
                if self.game.grid[row][col].is_visible and self.game.grid[row][col].bomb_count > 0]

    def generate_random_minefield(self, unknown_cells, num_mines):
        """ 生成一个可能的地雷分布 """
        if num_mines > len(unknown_cells):
            return set()
        return set(sample(unknown_cells, num_mines))

    def is_valid_minefield(self, minefield, constraint_cells):
        """ 检查地雷分布是否符合所有已知数字 """
        for row, col in constraint_cells:
            expected_count = self.game.grid[row][col].bomb_count
            neighbors = [(row + dx, col + dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1]
                         if (dx != 0 or dy != 0) and 0 <= row + dx < self.game.squares_y and 0 <= col + dy < self.game.squares_x]
            mine_count = sum(1 for neighbor in neighbors if neighbor in minefield)
            if mine_count != expected_count:
                return False
        return True

    def monte_carlo_analysis(self):
        """ 进行 Monte Carlo 采样，统计每个未知格子的地雷概率 """
        print("Monte Carlo 开始执行", flush=True)

        unknown_cells = self.get_unknown_cells()
        constraint_cells = self.get_constraint_cells()
        num_mines_remaining = self.game.num_bombs - sum(cell.has_bomb for row in self.game.grid for cell in row)

        print(f"未知格子: {len(unknown_cells)}, 剩余地雷: {num_mines_remaining}", flush=True)

        if not unknown_cells or num_mines_remaining <= 0:
            print("没有未知格子或剩余地雷数错误", flush=True)
            return None

        print("Monte Carlo 采样中...", flush=True)

        mine_prob = {cell: 0 for cell in unknown_cells}

        for i in range(self.simulations):
            if i % 100 == 0:
                print(f"采样进度: {i}/{self.simulations}", flush=True)
            try:
                minefield = self.generate_random_minefield(unknown_cells, num_mines_remaining)
                if self.is_valid_minefield(minefield, constraint_cells):
                    for cell in minefield:
                        mine_prob[cell] += 1
            except Exception as e:
                print(f"Monte Carlo 采样错误: {e}", flush=True)

        print("采样完成，计算最安全的格子", flush=True)

        safest_cell = min(mine_prob, key=mine_prob.get, default=None)
        print(f"选择最安全的格子: {safest_cell}，地雷概率: {mine_prob.get(safest_cell, '未知')}", flush=True)
        return safest_cell

    def make_best_move(self):
        """ AI 选择最安全的格子点击 """
        try:
            print("AI 开始计算最佳移动", flush=True)
            best_move = self.monte_carlo_analysis()

            # 如果 Monte Carlo 分析失败，AI 直接点击一个随机位置
            if best_move is None:
                unknown_cells = self.get_unknown_cells()
                if unknown_cells:
                    best_move = random.choice(unknown_cells)
                    print(f"没有足够信息，随机点击 {best_move}", flush=True)
                else:
                    print("没有未知格子，AI 终止运行", flush=True)
                    return

            if best_move:
                row, col = best_move
                print(f"AI 点击 ({row}, {col})", flush=True)
                self.game.click_handle(row, col, LEFT_CLICK)
                print(f"AI 已点击 ({row}, {col})", flush=True)
            else:
                print("AI 没有找到可点击的位置", flush=True)

        except Exception as e:
            print(f"AI 运行时发生错误: {e}", flush=True)


# 创建 AI Solver
solver = MonteCarloSolver(game)

# 让 AI 自动玩
if __name__ == "__main__":
    print("AI 启动，点击第一个随机格子...", flush=True)
    unknown_cells = solver.get_unknown_cells()
    if unknown_cells:
        first_click = random.choice(unknown_cells)
        print(f"AI 首次点击: {first_click}", flush=True)
        game.click_handle(first_click[0], first_click[1], LEFT_CLICK)

    while True:
        print("游戏主循环执行中...", flush=True)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:  
                print("退出游戏", flush=True)
                pygame.quit()
                sys.exit()

        print(f"游戏状态: game_lost={game.game_lost}, game_won={game.game_won}", flush=True)

        if not game.game_lost and not game.game_won:
            print("AI 运行 make_best_move()", flush=True)
            solver.make_best_move()

        game.draw()
        menu.draw(game)
        clock.tick(60)
        print("刷新游戏画面", flush=True)
        pygame.display.flip()
