import pygame
import sys
import random
import time
from minesweeper import Game, Menu, LEFT_CLICK, RIGHT_CLICK, NSQUARES_X, NSQUARES_Y, WIDTH, HEIGHT, MARGIN, MENU_SIZE

# --------------------------
# CSP Solver Class
# --------------------------
class CSPSolver:
    def __init__(self, game):
        self.game = game

    def initialize_fixed_game(self):
        """
        Uses `minesweeper.py` to create a fixed Minesweeper game.
        Ensures that a valid board is generated and precomputes constraints.
        """
        safe_x = random.randrange(self.game.squares_x)
        safe_y = random.randrange(self.game.squares_y)
        print(f"🛠 Generating fixed board. Safe start at ({safe_x}, {safe_y})")

        # Generate board using minesweeper.py logic
        self.game.place_bombs(safe_y, safe_x)
        self.game.count_all_bombs()
        self.game.click_handle(safe_y, safe_x, LEFT_CLICK)  # Start by revealing the safe cell

        # Update display after first move
        self.update_display()

    def update_display(self):
        """
        Ensures that the game board updates in real-time with bomb counts.
        """
        self.game.draw()
        pygame.display.flip()
        pygame.time.delay(100)

    def inference_step(self):
        """
        Apply CSP-based inference:
          - If a revealed cell’s bomb_count equals its flagged neighbors, all other unrevealed neighbors are safe.
          - If bomb_count == flagged_count + unrevealed neighbors, all must be bombs.
        """
        move_made = False

        for row in range(self.game.squares_y):
            for col in range(self.game.squares_x):
                cell = self.game.grid[row][col]

                if cell.is_visible and not cell.has_bomb and cell.bomb_count > 0:
                    unrevealed = []
                    flagged_count = 0

                    # Examine adjacent cells
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            r, c = row + dr, col + dc
                            if 0 <= r < self.game.squares_y and 0 <= c < self.game.squares_x:
                                neighbor = self.game.grid[r][c]
                                if neighbor.has_flag:
                                    flagged_count += 1
                                elif not neighbor.is_visible:
                                    unrevealed.append((r, c))

                    # Debugging Log: Display constraints being checked
                    print(f"🔍 Checking ({row},{col}) - Bomb Count: {cell.bomb_count}, Flagged: {flagged_count}, Unrevealed: {len(unrevealed)}")

                    # Inference 1: If flagged == bomb_count, all unrevealed must be safe.
                    if unrevealed and cell.bomb_count == flagged_count:
                        for (r, c) in unrevealed:
                            if not self.game.grid[r][c].is_visible and not self.game.grid[r][c].has_flag:
                                print(f"✅ Revealing safe cell at ({r}, {c}) based on cell ({row}, {col})")
                                self.game.click_handle(r, c, LEFT_CLICK)
                                move_made = True

                    # Inference 2: If bomb_count == flagged_count + unrevealed, all are bombs.
                    if unrevealed and cell.bomb_count == flagged_count + len(unrevealed):
                        for (r, c) in unrevealed:
                            if not self.game.grid[r][c].has_flag:
                                print(f"🚩 Flagging bomb at ({r}, {c}) based on cell ({row}, {col})")
                                self.game.click_handle(r, c, RIGHT_CLICK)
                                move_made = True

        if move_made:
            self.update_display()  # Update screen after making a move

        return move_made

    def propagate_inference(self):
        """Apply CSP logic continuously until no further deterministic moves exist."""
        while self.inference_step():
            self.update_display()

    def make_guess(self):
        """
        If no CSP-based moves are found, select an unrevealed cell adjacent to a revealed cell
        (a "frontier" cell) as the safest option. If no frontier exists, pick a random cell.
        """
        candidates = []
        for row in range(self.game.squares_y):
            for col in range(self.game.squares_x):
                cell = self.game.grid[row][col]
                if not cell.is_visible and not cell.has_flag:
                    # Check for at least one revealed neighbor
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            r, c = row + dr, col + dc
                            if 0 <= r < self.game.squares_y and 0 <= c < self.game.squares_x:
                                if self.game.grid[r][c].is_visible:
                                    candidates.append((row, col))
                                    break
                        else:
                            continue
                        break

        if not candidates:
            for row in range(self.game.squares_y):
                for col in range(self.game.squares_x):
                    if not self.game.grid[row][col].is_visible and not self.game.grid[row][col].has_flag:
                        candidates.append((row, col))

        if candidates:
            r, c = random.choice(candidates)
            print(f"⚠️ Guessing move at ({r}, {c})")
            self.game.click_handle(r, c, LEFT_CLICK)
            self.update_display()
            return True

        return False

    def solve_board(self):
        """
        Solves the Minesweeper board using CSP-based logic.
        """
        self.initialize_fixed_game()  # Generate a fixed board first

        while not self.game.game_won and not self.game.game_lost:
            self.propagate_inference()  # Apply CSP logic
            if not self.inference_step():
                self.make_guess()  # Guess if no CSP move available

# --------------------------
# Main Solver Loop
# --------------------------
def main():
    pygame.init()
    size = (NSQUARES_X * (WIDTH + MARGIN) + MARGIN,
            (NSQUARES_Y * (HEIGHT + MARGIN) + MARGIN) + MENU_SIZE)
    global screen
    screen = pygame.display.set_mode(size, pygame.RESIZABLE)
    pygame.display.set_caption("Minesweeper CSP Solver")
    global font
    font = pygame.font.Font('freesansbold.ttf', 24)
    
    game = Game()
    menu = Menu()
    global clock
    clock = pygame.time.Clock()

    solver = CSPSolver(game)
    solver.solve_board()  # Run the CSP solver

    # Final display: keep the window open until the user quits.
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        game.draw()
        menu.draw(game)
        pygame.display.flip()
        clock.tick(10)

if __name__ == "__main__":
    main()