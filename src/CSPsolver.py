"""
CSPsolver_sample.py

A sample (simplified) CSP solver for Minesweeper.
It builds a constraint model for the “frontier” cells (covered cells adjacent to a revealed number)
by treating each such cell as a boolean variable (0: safe, 1: mine) and collecting constraints
from the uncovered cells (each constraint says that the sum of adjacent mine‐variables equals the clue
minus any fixed (flagged) mines).

The solver uses recursive backtracking to enumerate all solutions, computes the probability that
each frontier cell is a mine, and chooses the cell with the lowest probability as the next move.

Usage (run this file directly):
    python CSPsolver_sample.py
This will open a Pygame Minesweeper window (using your existing minesweeper.py) and automatically
make moves based on the CSP solver.
"""

import pygame
import sys
from random import choice
from minesweeper import Game, Menu
from minesweeper import BLACK, WHITE, BLUE, RED, GRAY, MARGIN, WIDTH, HEIGHT, MENU_SIZE, LEFT_CLICK, RIGHT_CLICK

###############################################################################
# Simple CSP Solver for Minesweeper
###############################################################################

class SimpleMinesweeperCSP:
    def __init__(self, grid, rows, cols):
        """
        :param grid: 2D list of Cell objects (from minesweeper.py)
        :param rows: number of rows in the grid
        :param cols: number of columns in the grid
        """
        self.grid = grid
        self.rows = rows
        self.cols = cols

        # Frontier variables: covered cells adjacent to a revealed cell.
        # Domain for each: [0,1] (0 = safe, 1 = mine)
        self.variables = {}
        # Fixed assignments: cells that are not frontier (or are flagged)
        self.fixed = {}
        # Constraints: list of (target, [list of variable coordinates])
        self.constraints = []
        # Mapping from each variable to the indices of constraints in which it appears.
        self.var_to_constraints = {}
        
        self.build_csp()

    def build_csp(self):
        # Identify frontier cells (covered cells adjacent to a revealed cell)
        for i in range(self.rows):
            for j in range(self.cols):
                cell = self.grid[i][j]
                if not cell.is_visible:
                    adjacent_visible = False
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < self.rows and 0 <= nj < self.cols:
                                if self.grid[ni][nj].is_visible:
                                    adjacent_visible = True
                    if adjacent_visible:
                        if cell.has_flag:
                            self.fixed[(i, j)] = 1  # flagged cell is fixed as mine
                        else:
                            self.variables[(i, j)] = [0, 1]
                    else:
                        self.fixed[(i, j)] = 0  # not adjacent => safe
                        
        # Build constraints from each uncovered (visible) cell that is not a bomb
        for i in range(self.rows):
            for j in range(self.cols):
                cell = self.grid[i][j]
                if cell.is_visible and not cell.has_bomb:
                    target = cell.bomb_count
                    fixed_sum = 0
                    var_list = []
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < self.rows and 0 <= nj < self.cols:
                                neighbor = self.grid[ni][nj]
                                if not neighbor.is_visible:
                                    if (ni, nj) in self.fixed:
                                        fixed_sum += self.fixed[(ni, nj)]
                                    elif (ni, nj) in self.variables:
                                        var_list.append((ni, nj))
                    adjusted_target = target - fixed_sum
                    if var_list or adjusted_target != 0:
                        self.constraints.append((adjusted_target, var_list))
        
        # Build mapping: for each variable, record indices of constraints in which it appears
        self.var_to_constraints = {var: [] for var in self.variables}
        for idx, (target, var_list) in enumerate(self.constraints):
            for var in var_list:
                self.var_to_constraints[var].append(idx)

    def is_consistent(self, assignment, var):
        """
        Check consistency for constraints involving variable 'var'.
        For each constraint, ensure:
           sum(assigned values) <= target AND
           sum(assigned values) + (number of unassigned vars) >= target
        """
        for c_idx in self.var_to_constraints.get(var, []):
            target, var_list = self.constraints[c_idx]
            assigned_sum = 0
            unassigned = 0
            for v in var_list:
                if v in assignment:
                    assigned_sum += assignment[v]
                else:
                    unassigned += 1
            if assigned_sum > target or assigned_sum + unassigned < target:
                return False
        return True

    def _backtrack(self, assignment, var_list, idx, solutions):
        """
        Recursively enumerate all assignments for frontier variables that satisfy constraints.
        """
        if idx == len(var_list):
            solutions.append(dict(assignment))
            return
        var = var_list[idx]
        for value in self.variables[var]:
            assignment[var] = value
            if self.is_consistent(assignment, var):
                self._backtrack(assignment, var_list, idx + 1, solutions)
            del assignment[var]

    def find_all_solutions(self):
        """
        Return a list of all complete assignments (as dictionaries) that satisfy the CSP.
        """
        base_assignment = dict(self.fixed)
        var_list = list(self.variables.keys())
        solutions = []
        self._backtrack(base_assignment, var_list, 0, solutions)
        return solutions

    def compute_probabilities(self):
        """
        Compute for each frontier variable the probability of being a mine,
        based on the fraction of all solutions in which it is assigned 1.
        """
        solutions = self.find_all_solutions()
        if not solutions:
            return {}
        counts = {var: 0 for var in self.variables}
        for sol in solutions:
            for var in self.variables:
                counts[var] += sol.get(var, 0)
        probabilities = {var: counts[var] / len(solutions) for var in self.variables}
        return probabilities

    def choose_best_move(self):
        """
        Pick the frontier cell with the lowest probability of being a mine.
        If no CSP solutions are found, return a random covered, non-flagged cell.
        """
        probs = self.compute_probabilities()
        if not probs:
            # fallback: choose random frontier cell
            frontier = [(i, j) for i in range(self.rows) for j in range(self.cols)
                        if not self.grid[i][j].is_visible and not self.grid[i][j].has_flag]
            return choice(frontier) if frontier else None
        best_cell = min(probs, key=lambda v: probs[v])
        return best_cell

    def solve(self):
        """
        Return the coordinate (row, col) of the best move (lowest mine probability).
        """
        return self.choose_best_move()

###############################################################################
# Main Loop: Automatic CSP-Solver Demonstration in Pygame
###############################################################################

if __name__ == "__main__":
    pygame.init()
    game = Game()
    menu = Menu()
    clock = pygame.time.Clock()
    auto_solve = True  # Set to True to have the solver automatically make moves

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.VIDEORESIZE:
                if game.resize:
                    game.adjust_grid(event.w, event.h)
                    game.reset_game()
                else:
                    game.resize = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                col = pos[0] // (WIDTH + MARGIN)
                row = (pos[1] - MENU_SIZE) // (HEIGHT + MARGIN)
                if row >= game.squares_y:
                    row = game.squares_y - 1
                if col >= game.squares_x:
                    col = game.squares_x - 1
                if row >= 0:
                    game.click_handle(row, col, event.button)
                else:
                    menu.click_handle(game)

        # Automatic move: if the game is ongoing, use the CSP solver to pick the best move
        if auto_solve and not game.game_lost and not game.game_won:
            csp_solver = SimpleMinesweeperCSP(game.grid, game.squares_y, game.squares_x)
            move = csp_solver.solve()
            if move is not None:
                r, c = move
                if not game.grid[r][c].is_visible:
                    game.click_handle(r, c, LEFT_CLICK)
                    pygame.time.delay(500)  # Delay to visually observe the move

        game.draw()
        menu.draw(game)
        clock.tick(10)
        pygame.display.flip()