"""
CSPsolver_sample.py

An advanced and memory–efficient CSP solver for Minesweeper that fully leverages the
information computed by minesweeper.py. In this version:
  • Forced–move propagation is applied using the already–computed bomb counts,
    revealing safe cells or flagging forced mines.
  • The CSP model is built only over the “frontier” (covered cells adjacent to a revealed cell)
    and uses forward–checking to reduce domains.
  • Dynamic variable ordering is applied (MRV combined with degree heuristic).
  • When choosing among candidate moves, ties (equal mine probability) are broken by favoring
    cells with more adjacent revealed cells (thus likely to yield more new information).
  • If no CSP solution is found, the solver falls back on a random guess among all remaining
    covered cells.
    
The board is drawn as usual (revealed cells show numbers like 1,2,3,…) and a debug panel in the menu
displays the number of frontier cells.

Usage:
    python CSPsolver_sample.py
This will open a Pygame Minesweeper window (using your existing minesweeper.py) and automatically play
the game using this improved CSP method.
"""

import pygame
import sys
from random import choice
from collections import deque

from minesweeper import Game, Menu
from minesweeper import BLACK, WHITE, BLUE, RED, GRAY, MARGIN, WIDTH, HEIGHT, MENU_SIZE, LEFT_CLICK, RIGHT_CLICK

###############################################################################
# Forced Move Propagation (using existing revealed clues)
###############################################################################
def propagate_forced_moves(game):
    """
    Scan all revealed cells and, using their bomb_count, deduce forced moves:
      - If bomb_count equals the number of flagged neighbors, then all other covered neighbors are safe.
      - If bomb_count equals flagged count plus covered neighbors, then all covered neighbors are forced mines.
    Returns True if at least one move was made.
    """
    move_made = False
    for i in range(game.squares_y):
        for j in range(game.squares_x):
            cell = game.grid[i][j]
            if cell.is_visible and not cell.has_bomb:
                flagged = 0
                covered = []
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < game.squares_y and 0 <= nj < game.squares_x:
                            neighbor = game.grid[ni][nj]
                            if neighbor.has_flag:
                                flagged += 1
                            elif not neighbor.is_visible:
                                covered.append((ni, nj))
                if not covered:
                    continue
                if cell.bomb_count == flagged:
                    for (ni, nj) in covered:
                        if not game.grid[ni][nj].is_visible:
                            game.click_handle(ni, nj, LEFT_CLICK)
                            move_made = True
                elif cell.bomb_count == flagged + len(covered):
                    for (ni, nj) in covered:
                        if not game.grid[ni][nj].has_flag:
                            game.click_handle(ni, nj, RIGHT_CLICK)
                            move_made = True
    return move_made

###############################################################################
# Debug Panel: Display Additional Information
###############################################################################
def draw_debug_info(game, surface, font):
    """
    Draw debug information (e.g., number of frontier cells) in the upper-right corner.
    """
    frontier = []
    for i in range(game.squares_y):
        for j in range(game.squares_x):
            if not game.grid[i][j].is_visible and not game.grid[i][j].has_flag:
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < game.squares_y and 0 <= nj < game.squares_x:
                            if game.grid[ni][nj].is_visible:
                                frontier.append((i, j))
                                break
                    else:
                        continue
                    break
    debug_text = f"Frontier: {len(frontier)}"
    text_surface = font.render(debug_text, True, BLUE)
    surface.blit(text_surface, (game.squares_x*(WIDTH+MARGIN) - 150, 10))

###############################################################################
# CSP Solver: Using Existing Information with Forward Checking & MRV
###############################################################################
class SimpleMinesweeperCSP:
    def __init__(self, grid, rows, cols):
        """
        Build a constraint model for the remaining frontier cells using the grid's existing info.
        :param grid: 2D list of Cell objects.
        :param rows: number of rows.
        :param cols: number of columns.
        """
        self.grid = grid
        self.rows = rows
        self.cols = cols
        self.variables = {}  # frontier cells: domain (list of possible values)
        self.fixed = {}      # cells fixed as 0 or 1 (non-frontier or flagged)
        self.constraints = []  # List of (target, [list of variable coords])
        self.var_to_constraints = {}
        self.build_csp()

    def build_csp(self):
        # Use the game’s already–computed info. A covered cell is in the frontier if it neighbors any revealed cell.
        for i in range(self.rows):
            for j in range(self.cols):
                cell = self.grid[i][j]
                if not cell.is_visible:
                    adjacent = False
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < self.rows and 0 <= nj < self.cols:
                                if self.grid[ni][nj].is_visible:
                                    adjacent = True
                    if adjacent:
                        if cell.has_flag:
                            self.fixed[(i, j)] = 1
                        else:
                            # Domain starts as [0,1]
                            self.variables[(i, j)] = [0, 1]
                    else:
                        self.fixed[(i, j)] = 0

        # Build constraints from every revealed cell that is not a bomb.
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
        # Map each variable to constraints in which it appears.
        self.var_to_constraints = {var: [] for var in self.variables}
        for idx, (target, var_list) in enumerate(self.constraints):
            for var in var_list:
                self.var_to_constraints[var].append(idx)

        # (Forward checking) You might later reduce domains here if you detect any forced singleton.
        self.forward_propagate()

    def forward_propagate(self):
        """
        For every variable, check if any constraint forces its domain to a singleton.
        Update the variable's domain if possible.
        (For our binary domain [0,1], if a constraint forces a variable to be 0 or 1, then update it.)
        """
        changed = True
        while changed:
            changed = False
            for var in list(self.variables.keys()):
                # For each constraint involving var, narrow its domain.
                new_domain = self.variables[var][:]
                for c_idx in self.var_to_constraints.get(var, []):
                    target, var_list = self.constraints[c_idx]
                    # For constraint, determine min and max possible sum over variables other than var.
                    other_vars = [v for v in var_list if v != var]
                    min_other = sum(0 for _ in other_vars)
                    max_other = sum(1 for _ in other_vars)
                    # Then var must be in a range so that: value + min_other <= target <= value + max_other.
                    possible = []
                    for val in new_domain:
                        if val + min_other <= target <= val + max_other:
                            possible.append(val)
                    if possible != new_domain:
                        new_domain = possible
                        changed = True
                # Update if reduced.
                if len(new_domain) < len(self.variables[var]):
                    self.variables[var] = new_domain
                    # If domain is a singleton, add it to fixed.
                    if len(new_domain) == 1:
                        self.fixed[var] = new_domain[0]
                        del self.variables[var]
        # Rebuild var_to_constraints without fixed variables.
        self.var_to_constraints = {var: cons for var, cons in self.var_to_constraints.items() if var in self.variables}

    def is_consistent(self, assignment, var):
        for c_idx in self.var_to_constraints.get(var, []):
            target, var_list = self.constraints[c_idx]
            assigned = sum(assignment.get(v, 0) for v in var_list)
            unassigned = sum(1 for v in var_list if v not in assignment)
            if assigned > target or assigned + unassigned < target:
                return False
        return True

    def _backtrack(self, assignment, var_list, idx, solutions, memo):
        if idx == len(var_list):
            solutions.append(dict(assignment))
            return
        key = (idx, tuple(sorted(assignment.items())))
        if key in memo:
            return
        # Use MRV: sort remaining variables by current domain size then by degree.
        remaining = sorted(var_list[idx:], key=lambda v: (len(self.variables[v]), -len(self.var_to_constraints.get(v, []))))
        # Swap the first remaining variable to the current index.
        var = remaining[0]
        index_in_list = var_list.index(var)
        var_list[idx], var_list[index_in_list] = var_list[index_in_list], var_list[idx]
        for value in self.variables[var]:
            assignment[var] = value
            if self.is_consistent(assignment, var):
                self._backtrack(assignment, var_list, idx+1, solutions, memo)
            del assignment[var]
        memo.add(key)

    def find_all_solutions(self):
        base_assignment = dict(self.fixed)
        var_list = list(self.variables.keys())
        solutions = []
        memo = set()
        self._backtrack(base_assignment, var_list, 0, solutions, memo)
        return solutions

    def compute_probabilities(self):
        sols = self.find_all_solutions()
        if not sols:
            return {}
        counts = {var: 0 for var in self.variables}
        for sol in sols:
            for var in self.variables:
                counts[var] += sol.get(var, 0)
        probabilities = {var: counts[var] / len(sols) for var in self.variables}
        return probabilities

    def choose_best_move(self, game):
        probs = self.compute_probabilities()
        if not probs:
            # Fallback: choose a random covered, unflagged cell.
            all_cells = [(i, j) for i in range(self.rows) for j in range(self.cols)
                         if not self.grid[i][j].is_visible and not self.grid[i][j].has_flag]
            return choice(all_cells) if all_cells else None
        # If any frontier cell is forced safe, choose it.
        for var, p in probs.items():
            if p == 0:
                return var
        # If any cell is forced mine, flag it.
        for var, p in probs.items():
            if p == 1:
                r, c = var
                if not self.grid[r][c].has_flag:
                    game.click_handle(r, c, RIGHT_CLICK)
                probs[var] = 1.0
        # Break ties: choose cell with lowest probability, but among equals pick the one with most adjacent revealed cells.
        best = None
        best_prob = 1.1
        best_neighbors = -1
        for var, p in probs.items():
            if p < best_prob:
                best = var
                best_prob = p
                best_neighbors = count_revealed_neighbors(game, var)
            elif abs(p - best_prob) < 1e-6:  # tie
                neighbor_count = count_revealed_neighbors(game, var)
                if neighbor_count > best_neighbors:
                    best = var
                    best_neighbors = neighbor_count
        return best

    def solve(self, game):
        return self.choose_best_move(game)

def count_revealed_neighbors(game, var):
    i, j = var
    count = 0
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            ni, nj = i+di, j+dj
            if 0 <= ni < game.squares_y and 0 <= nj < game.squares_x:
                if game.grid[ni][nj].is_visible:
                    count += 1
    return count

###############################################################################
# Overall Solver Loop: Propagation, CSP, and Iteration Until Solved
###############################################################################
def solve_board(game):
    """
    Repeatedly:
      1. Apply forced–move propagation.
      2. Build the CSP model (using existing information) and use forward checking.
      3. If a safe move is deduced (or a forced mine is flagged), execute it;
         otherwise, choose the move with the lowest estimated mine probability.
    """
    # Propagate forced moves until no further move is made.
    while propagate_forced_moves(game):
        game.draw()
        pygame.display.flip()
        pygame.time.delay(100)
    # Build and solve the CSP model.
    csp_solver = SimpleMinesweeperCSP(game.grid, game.squares_y, game.squares_x)
    move = csp_solver.solve(game)
    if move is not None:
        r, c = move
        if not game.grid[r][c].is_visible:
            game.click_handle(r, c, LEFT_CLICK)
            pygame.time.delay(200)
    return

###############################################################################
# Main Loop: Automatic Solver Demonstration in Pygame
###############################################################################
if __name__ == "__main__":
    pygame.init()
    game = Game()
    menu = Menu()
    clock = pygame.time.Clock()
    auto_solve = True
    debug_font = pygame.font.Font('freesansbold.ttf', 16)

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

        if auto_solve and not game.game_lost and not game.game_won:
            solve_board(game)

        game.draw()
        menu.draw(game)
        draw_debug_info(game, pygame.display.get_surface(), debug_font)
        clock.tick(10)
        pygame.display.flip()