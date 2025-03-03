import pygame
import sys
from random import choice
# Import your Minesweeper environment. Ensure minesweeper.py is in your project directory.
from minesweeper import Game, Menu, BLACK, WHITE, BLUE, RED, GRAY, MARGIN, WIDTH, HEIGHT, MENU_SIZE, LEFT_CLICK, RIGHT_CLICK

class DynamicProgramming:
    def __init__(self, game, theta=1e-6, max_iterations=1000):
        """
        Initialize Dynamic Programming solver for Minesweeper environment.
        
        Args:
            game (Game): Instance of the Minesweeper game environment.
            theta (float): Convergence threshold (unused here but provided for format).
            max_iterations (int): Maximum iterations for DP recursion.
        """
        self.game = game
        self.theta = theta
        self.max_iterations = max_iterations

    def extract_frontier_and_constraints(self):
        """
        Extract the frontier (unrevealed cells adjacent to a revealed numbered cell)
        and constraints from the game board.

        Returns:
            frontier (list): List of tuples (row, col) representing frontier cells.
            constraints (list): List of tuples (revealed_cell, [adjacent unknown cells], bombs_remaining)
                                where bombs_remaining = cell.bomb_count - number of adjacent flagged cells.
        """
        frontier = set()
        constraints = []
        for r in range(self.game.squares_y):
            for c in range(self.game.squares_x):
                cell = self.game.grid[r][c]
                if cell.is_visible and cell.bomb_count > 0:
                    unknown_neighbors = []
                    flagged_count = 0
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            nr = r + dr
                            nc = c + dc
                            if 0 <= nr < self.game.squares_y and 0 <= nc < self.game.squares_x:
                                neighbor = self.game.grid[nr][nc]
                                if not neighbor.is_visible and not neighbor.has_flag:
                                    unknown_neighbors.append((nr, nc))
                                elif neighbor.has_flag:
                                    flagged_count += 1
                    if unknown_neighbors:
                        required = cell.bomb_count - flagged_count
                        if 0 <= required <= len(unknown_neighbors):
                            constraints.append(((r, c), unknown_neighbors, required))
                            frontier.update(unknown_neighbors)
        return list(frontier), constraints

    def solve_dp(self, frontier, constraints):
        """
        Use recursive dynamic programming with memoization to count valid bomb assignments
        for the frontier cells, and compute for each cell the number of configurations
        in which it is a bomb.
        
        Args:
            frontier (list): List of frontier cell tuples.
            constraints (list): List of constraints as returned by extract_frontier_and_constraints.
        
        Returns:
            probabilities (dict): Mapping from frontier cell (row, col) to probability of bomb.
        """
        frontier = sorted(frontier)
        n = len(frontier)

        # Build constraint indices: for each constraint, record indices in the frontier that are involved.
        cons_indices = []
        cons_required = []
        for cons in constraints:
            (_, cells, req) = cons
            indices = []
            for cell in cells:
                if cell in frontier:
                    indices.append(frontier.index(cell))
            cons_indices.append(indices)
            cons_required.append(req)

        memo = {}

        def dp(index, req_tuple):
            if index == n:
                if all(x == 0 for x in req_tuple):
                    return 1, [0] * n
                else:
                    return 0, [0] * n

            key = (index, req_tuple)
            if key in memo:
                return memo[key]

            total_configs = 0
            bomb_counts = [0] * n

            # Option 1: assign a bomb at frontier[index]
            new_req = list(req_tuple)
            valid = True
            for i, indices in enumerate(cons_indices):
                if index in indices:
                    new_req[i] -= 1
                    if new_req[i] < 0:
                        valid = False
                        break
            if valid:
                count_true, bomb_counts_true = dp(index + 1, tuple(new_req))
                total_configs += count_true
                bomb_counts[index] += count_true  # Current cell is a bomb in these configurations
                bomb_counts = [bomb_counts[i] + bomb_counts_true[i] for i in range(n)]
            
            # Option 2: do not assign a bomb at frontier[index]
            count_false, bomb_counts_false = dp(index + 1, req_tuple)
            total_configs += count_false
            bomb_counts = [bomb_counts[i] + bomb_counts_false[i] for i in range(n)]

            memo[key] = (total_configs, bomb_counts)
            return memo[key]

        total, bomb_counts = dp(0, tuple(cons_required))
        
        # If no valid configuration is found, fallback to selecting a random frontier cell.
        if total == 0:
            random_cell = choice(frontier)
            return {cell: (1.0 if cell == random_cell else 0.0) for cell in frontier}
        
        probabilities = {}
        for i, cell in enumerate(frontier):
            probabilities[cell] = bomb_counts[i] / total
        return probabilities

    def solve(self):
        """
        Determine the best move using the DP solver.
        If the game hasn't started (first move), choose a predetermined initial move.
        Otherwise, compute the frontier and return the cell with the lowest bomb probability.
        
        Returns:
            best_move (tuple): (row, col) of the cell to click.
        """
        # If game hasn't started, choose an initial move.
        if not self.game.init:
            center_row = self.game.squares_y // 2
            center_col = self.game.squares_x // 2
            self.game.init = True  # Mark the game as initialized.
            return (center_row, center_col)
        
        frontier, constraints = self.extract_frontier_and_constraints()
        hidden = [(r, c) for r in range(self.game.squares_y)
                          for c in range(self.game.squares_x)
                          if not self.game.grid[r][c].is_visible and not self.game.grid[r][c].has_flag]
        
        # If no frontier or constraints, fall back to a random hidden cell.
        if not frontier or not constraints:
            return choice(hidden) if hidden else None

        probabilities = self.solve_dp(frontier, constraints)
        if not probabilities:
            return choice(hidden) if hidden else None

        best_move = min(probabilities, key=probabilities.get)
        return best_move

# -------------------------------
# Main Loop Integrating the DP Solver
# -------------------------------
if __name__ == "__main__":
    pygame.init()
    # Initialize the Minesweeper game and menu.
    game = Game()
    menu = Menu()
    dp_solver = DynamicProgramming(game)
    clock = pygame.time.Clock()
    auto_solve = True  # Set to True to allow the solver to make moves automatically

    while True:
        for event in pygame.event.get():
            # Closes the game if user clicks the X
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            # Handle screen resize events
            elif event.type == pygame.VIDEORESIZE:
                if game.resize:
                    game.adjust_grid(event.w, event.h)
                    game.reset_game()
                else:
                    game.resize = True
            # Handle mouse clicks
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                column = pos[0] // (WIDTH + MARGIN)
                row = (pos[1] - MENU_SIZE) // (HEIGHT + MARGIN)
                if row >= game.squares_y:
                    row = game.squares_y - 1
                if column >= game.squares_x:
                    column = game.squares_x - 1
                if row >= 0:
                    game.click_handle(row, column, event.button)
                else:
                    menu.click_handle(game)

        # Auto-solver move: if the game is in progress, let the DP solver choose a move.
        if auto_solve and not game.game_lost and not game.game_won:
            best_move = dp_solver.solve()
            if best_move is not None:
                r, c = best_move
                if not game.grid[r][c].is_visible:
                    game.click_handle(r, c, LEFT_CLICK)
                    pygame.time.delay(500)  # Delay to allow board state to update visibly

        game.draw()
        menu.draw(game)
        clock.tick(10)  # Adjust frame rate as needed
        pygame.display.flip()
