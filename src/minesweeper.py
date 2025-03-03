import pygame
import sys
import random

# Initialize pygame
pygame.init()

# Game settings for Expert level (16 rows x 30 columns)
ROWS, COLS = 16, 30
CELL_SIZE = 30
WIDTH, HEIGHT = COLS * CELL_SIZE, ROWS * CELL_SIZE  # 900x480
MINES_COUNT = 99

# Classic Minesweeper colors
UNREVEALED_COLOR = (192, 192, 192)  # classic gray for unrevealed cells
REVEALED_COLOR = (255, 255, 255)    # white for revealed cells
GRID_COLOR = (128, 128, 128)         # darker gray for grid lines
FLAG_COLOR = (255, 0, 0)             # red for flags
MINE_COLOR = (0, 0, 0)               # black for mines

# Mapping of numbers to classic Minesweeper colors
NUM_COLORS = {
    1: (0, 0, 255),      # blue
    2: (0, 128, 0),      # green
    3: (255, 0, 0),      # red
    4: (0, 0, 128),      # dark blue
    5: (128, 0, 0),      # maroon
    6: (0, 128, 128),    # teal
    7: (0, 0, 0),        # black
    8: (128, 128, 128),  # gray
}

# Create screen with adjusted dimensions
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Minesweeper Expert")

# Font for numbers
font = pygame.font.SysFont(None, 30)

class Cell:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.x = col * CELL_SIZE
        self.y = row * CELL_SIZE
        self.is_mine = False
        self.is_revealed = False
        self.is_flagged = False
        self.neighbor_mines = 0

    def draw(self, surface):
        rect = pygame.Rect(self.x, self.y, CELL_SIZE, CELL_SIZE)
        # Draw cell background
        if self.is_revealed:
            pygame.draw.rect(surface, REVEALED_COLOR, rect)
        else:
            pygame.draw.rect(surface, UNREVEALED_COLOR, rect)
        
        # Draw grid lines
        pygame.draw.rect(surface, GRID_COLOR, rect, 1)

        # Draw flag on unrevealed cells
        if self.is_flagged:
            pygame.draw.circle(surface, FLAG_COLOR, rect.center, CELL_SIZE // 4)
        
        # Draw mine or number if revealed
        if self.is_revealed:
            if self.is_mine:
                pygame.draw.circle(surface, MINE_COLOR, rect.center, CELL_SIZE // 3)
            elif self.neighbor_mines > 0:
                text_color = NUM_COLORS.get(self.neighbor_mines, (0, 0, 0))
                text = font.render(str(self.neighbor_mines), True, text_color)
                text_rect = text.get_rect(center=rect.center)
                surface.blit(text, text_rect)

def create_grid():
    grid = [[Cell(r, c) for c in range(COLS)] for r in range(ROWS)]
    return grid

def place_mines(grid, initial_click=None):
    # Create a list of all cell positions
    all_cells = [(r, c) for r in range(ROWS) for c in range(COLS)]
    # Optionally avoid placing a mine on the initial click or its neighbors
    if initial_click:
        r0, c0 = initial_click
        avoid = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = r0 + dr, c0 + dc
                if 0 <= nr < ROWS and 0 <= nc < COLS:
                    avoid.append((nr, nc))
        available = [pos for pos in all_cells if pos not in avoid]
    else:
        available = all_cells

    mines = random.sample(available, MINES_COUNT)
    for (r, c) in mines:
        grid[r][c].is_mine = True

def count_neighbor_mines(grid):
    for r in range(ROWS):
        for c in range(COLS):
            if grid[r][c].is_mine:
                continue
            count = 0
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < ROWS and 0 <= nc < COLS:
                        if grid[nr][nc].is_mine:
                            count += 1
            grid[r][c].neighbor_mines = count

def reveal_cell(grid, row, col):
    if row < 0 or row >= ROWS or col < 0 or col >= COLS:
        return
    cell = grid[row][col]
    if cell.is_revealed or cell.is_flagged:
        return

    cell.is_revealed = True
    # Flood-fill: if there are no adjacent mines, reveal neighboring cells
    if cell.neighbor_mines == 0 and not cell.is_mine:
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                reveal_cell(grid, row + dr, col + dc)

def check_win(grid):
    for row in grid:
        for cell in row:
            if not cell.is_mine and not cell.is_revealed:
                return False
    return True

def game_loop():
    grid = create_grid()
    mines_placed = False
    game_over = False
    win = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and not game_over:
                mouse_pos = pygame.mouse.get_pos()
                col = mouse_pos[0] // CELL_SIZE
                row = mouse_pos[1] // CELL_SIZE
                cell = grid[row][col]

                # Left click to reveal cell
                if event.button == 1:
                    if not mines_placed:
                        place_mines(grid, initial_click=(row, col))
                        count_neighbor_mines(grid)
                        mines_placed = True

                    if not cell.is_flagged:
                        cell.is_revealed = True
                        if cell.is_mine:
                            game_over = True
                        elif cell.neighbor_mines == 0:
                            reveal_cell(grid, row, col)
                    if check_win(grid):
                        win = True
                        game_over = True

                # Right click to toggle flag
                if event.button == 3:
                    if not cell.is_revealed:
                        cell.is_flagged = not cell.is_flagged

        # Drawing the grid
        screen.fill(UNREVEALED_COLOR)
        for row in grid:
            for cell in row:
                cell.draw(screen)

        pygame.display.flip()

        # Optionally display game over message
        if game_over:
            msg = "You Win!" if win else "Game Over!"
            print(msg)
            pygame.time.wait(2000)
            return

if __name__ == "__main__":
    while True:
        game_loop()
