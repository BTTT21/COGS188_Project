import pygame
import sys
import random
from random import randrange
from functools import lru_cache

# -------------------- Constants and Colors --------------------
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE  = (0, 0, 255)
GREEN = (0, 255, 0)
RED   = (255, 0, 0)
GRAY  = (127, 127, 127)

WIDTH = 30       # width of each cell
HEIGHT = 30      # height of each cell
MARGIN = 5       # margin between cells
MENU_SIZE = 40   # height of the menu bar
LEFT_CLICK = 1
RIGHT_CLICK = 3

# Expert board settings
NSQUARES_X = 16  # columns
NSQUARES_Y = 16  # rows
EXPERT_BOMBS = 40

# -------------------- Game Class --------------------
class Game:
    def __init__(self):
        self.squares_x = NSQUARES_X
        self.squares_y = NSQUARES_Y
        self.grid = [[self.Cell(x, y) for x in range(self.squares_x)] for y in range(self.squares_y)]
        self.init = False
        self.game_lost = False
        self.game_won = False
        self.num_bombs = EXPERT_BOMBS
        self.resize = False
        self.flag_count = 0

    def draw(self):
        screen.fill(BLACK)
        for row in range(self.squares_y):
            for column in range(self.squares_x):
                color = WHITE
                if self.grid[row][column].is_visible:
                    color = RED if self.grid[row][column].has_bomb else GRAY
                elif self.grid[row][column].has_flag:
                    color = BLUE
                pygame.draw.rect(screen,
                                 color,
                                 [(MARGIN + WIDTH) * column + MARGIN,
                                  (MARGIN + HEIGHT) * row + MARGIN + MENU_SIZE,
                                  WIDTH,
                                  HEIGHT])
                self.grid[row][column].show_text()

    def adjust_grid(self, sizex, sizey):
        global screen
        self.squares_x = (sizex - MARGIN) // (WIDTH + MARGIN)
        self.squares_y = (sizey - MARGIN - MENU_SIZE) // (HEIGHT + MARGIN)
        if self.squares_x < 8:
            self.squares_x = 8
        if self.squares_y < 8:
            self.squares_y = 8
        if self.num_bombs > (self.squares_x * self.squares_y) // 3:
            self.num_bombs = (self.squares_x * self.squares_y) // 3
        self.grid = [[self.Cell(x, y) for x in range(self.squares_x)] for y in range(self.squares_y)]
        size = ((self.squares_x * (WIDTH + MARGIN) + MARGIN),
                (self.squares_y * (HEIGHT + MARGIN) + MARGIN + MENU_SIZE))
        screen = pygame.display.set_mode(size, pygame.RESIZABLE)

    def game_over(self):
        for row in range(self.squares_y):
            for column in range(self.squares_x):
                if self.grid[row][column].has_bomb:
                    self.grid[row][column].is_visible = True
                self.grid[row][column].has_flag = False

    def change_num_bombs(self, bombs):
        self.num_bombs += bombs
        if self.num_bombs < 1:
            self.num_bombs = 1
        elif self.num_bombs > (self.squares_x * self.squares_y) // 3:
            self.num_bombs = (self.squares_x * self.squares_y) // 3
        self.reset_game()

    def place_bombs(self, row, column):
        bombplaced = 0
        while bombplaced < self.num_bombs:
            x = randrange(self.squares_y)
            y = randrange(self.squares_x)
            if not self.grid[x][y].has_bomb:
                self.grid[x][y].has_bomb = True
                bombplaced += 1
        self.count_all_bombs()

    def count_all_bombs(self):
        for row in range(self.squares_y):
            for column in range(self.squares_x):
                self.grid[row][column].count_bombs(self.squares_y, self.squares_x)

    def reset_game(self):
        for row in range(self.squares_y):
            for column in range(self.squares_x):
                self.init = False
                self.grid[row][column].is_visible = False
                self.grid[row][column].has_bomb = False
                self.grid[row][column].bomb_count = 0
                self.grid[row][column].test = False
                self.grid[row][column].has_flag = False
        self.game_lost = False
        self.game_won = False
        self.flag_count = 0

    def check_victory(self):
        count = 0
        total = self.squares_x * self.squares_y
        for row in range(self.squares_y):
            for column in range(self.squares_x):
                if self.grid[row][column].is_visible:
                    count += 1
        # If the only cells left hidden are bombs, then we've won
        if ((total - count) == self.num_bombs) and not self.game_lost:
            self.game_won = True
            for row in range(self.squares_y):
                for column in range(self.squares_x):
                    if self.grid[row][column].has_bomb:
                        self.grid[row][column].has_flag = True

    def count_flags(self):
        total_flags = 0
        for row in range(self.squares_y):
            for column in range(self.squares_x):
                if self.grid[row][column].has_flag:
                    total_flags += 1
        self.flag_count = total_flags

    def click_handle(self, row, column, button):
        if button == LEFT_CLICK and self.game_won:
            self.reset_game()
        elif button == LEFT_CLICK and not self.grid[row][column].has_flag:
            if not self.game_lost:
                if not self.init:
                    self.place_bombs(row, column)
                    self.init = True
                self.grid[row][column].is_visible = True
                self.grid[row][column].has_flag = False
                if self.grid[row][column].has_bomb:
                    self.game_over()
                    self.game_lost = True
                if self.grid[row][column].bomb_count == 0 and not self.grid[row][column].has_bomb:
                    self.grid[row][column].open_neighbours(self.squares_y, self.squares_x)
                self.check_victory()
            else:
                # If we click while the game is lost, reset
                self.game_lost = False
                self.reset_game()
        elif button == RIGHT_CLICK and not self.game_won:
            # Toggle flag
            if not self.grid[row][column].has_flag:
                if self.flag_count < self.num_bombs and not self.grid[row][column].is_visible:
                    self.grid[row][column].has_flag = True
            else:
                self.grid[row][column].has_flag = False
            self.count_flags()

    class Cell:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.is_visible = False
            self.has_bomb = False
            self.bomb_count = 0
            self.text = ""
            self.test = False
            self.has_flag = False

        def show_text(self):
            if self.is_visible:
                if self.bomb_count == 0:
                    self.text = font.render("", True, BLACK)
                else:
                    self.text = font.render(str(self.bomb_count), True, BLACK)
                screen.blit(self.text, (self.x * (WIDTH + MARGIN) + 12,
                                        self.y * (HEIGHT + MARGIN) + 10 + MENU_SIZE))

        def count_bombs(self, max_rows, max_cols):
            if not self.test:
                self.test = True
                if not self.has_bomb:
                    for col in range(self.x - 1, self.x + 2):
                        for row in range(self.y - 1, self.y + 2):
                            if (row >= 0 and row < max_rows and
                                col >= 0 and col < max_cols and
                                not (col == self.x and row == self.y) and
                                game.grid[row][col].has_bomb):
                                self.bomb_count += 1

        def open_neighbours(self, max_rows, max_cols):
            col = self.x
            row = self.y
            for row_off in range(-1, 2):
                for col_off in range(-1, 2):
                    # Note: This version opens only orthogonal neighbors
                    if ((row_off == 0 or col_off == 0) and row_off != col_off and
                        row + row_off >= 0 and col + col_off >= 0 and
                        row + row_off < max_rows and col + col_off < max_cols):
                        cell = game.grid[row + row_off][col + col_off]
                        cell.count_bombs(game.squares_y, game.squares_x)
                        if (not cell.is_visible and not cell.has_bomb):
                            cell.is_visible = True
                            cell.has_flag = False
                            if cell.bomb_count == 0:
                                cell.open_neighbours(game.squares_y, game.squares_x)

# -------------------- Menu Class --------------------
class Menu:
    def __init__(self):
        self.width = pygame.display.get_surface().get_width() - 2 * MARGIN
        self.btn_minus = self.Button(10, 10, 20, 20, "-", 6, -3)
        self.btn_plus = self.Button(60, 10, 20, 20, "+", 3, -4)
        self.btn_flags = self.Button(280, 16, 10, 10, "")
        self.btn_flags.background = BLUE
        self.label_bombs = self.Label(30, 10)
        self.label_game_end = self.Label(100, 10)
        self.label_flags = self.Label(self.width - 50, 10)

    def click_handle(self, obj):
        if self.btn_minus.click_handle():
            obj.change_num_bombs(-1)
        if self.btn_plus.click_handle():
            obj.change_num_bombs(1)

    def draw(self, obj):
        self.width = pygame.display.get_surface().get_width() - 2 * MARGIN 
        pygame.draw.rect(screen, GRAY, [MARGIN, 0, self.width, MENU_SIZE])
        self.btn_minus.draw(screen)
        self.btn_plus.draw(screen)
        self.btn_flags.draw(screen)
        self.label_bombs.show(screen, game.num_bombs)
        self.label_flags.show(screen, game.flag_count)
        if obj.game_lost:
            self.label_game_end.show(screen, "Game Over")
        elif obj.game_won:
            self.label_game_end.show(screen, "You Won!")

    class Label:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.text = ""
        
        def show(self, surface, value):
            text = str(value)
            self.text = font.render(text, True, BLACK)
            surface.blit(self.text, (self.x, self.y))

    class Button:
        def __init__(self, x, y, width, height, text, xoff=0, yoff=0):
            self.x = x
            self.y = y
            self.height = height
            self.width = width
            self.background = WHITE
            self.text = text
            self.x_offset = xoff
            self.y_offset = yoff

        def draw(self, surface):
            pygame.draw.ellipse(surface, self.background, [self.x, self.y, self.width, self.height], 0)
            text = font.render(self.text, True, BLACK)
            surface.blit(text, (self.x + self.x_offset, self.y + self.y_offset))

        def click_handle(self):
            pos = pygame.mouse.get_pos()
            if (self.x < pos[0] < self.x + self.width) and (self.y < pos[1] < self.y + self.height):
                return True
            return False

# -------------------- CSP Solver Utilities --------------------
def get_neighbors(r, c, max_rows, max_cols):
    neighbors = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < max_rows and 0 <= nc < max_cols:
                neighbors.append((nr, nc))
    return neighbors

def get_frontier_cells(game):
    """
    Frontier = set of hidden/unflagged cells that are adjacent 
               to at least one revealed cell with bomb_count > 0.
    """
    frontier = set()
    for r in range(game.squares_y):
        for c in range(game.squares_x):
            cell = game.grid[r][c]
            if not cell.is_visible and not cell.has_flag:
                # Check if any neighbor is a visible clue with bomb_count>0
                for nr, nc in get_neighbors(r, c, game.squares_y, game.squares_x):
                    neighbor = game.grid[nr][nc]
                    if neighbor.is_visible and neighbor.bomb_count > 0:
                        frontier.add((r, c))
                        break
    return frontier

def get_constraints(game, frontier):
    """
    Constraints: For each revealed cell with bomb_count > 0,
    how many bombs among its adjacency are in the frontier?
    (Also accounting for flagged cells).
    """
    constraints = {}
    for r in range(game.squares_y):
        for c in range(game.squares_x):
            cell = game.grid[r][c]
            if cell.is_visible and cell.bomb_count > 0:
                adj_frontier = []
                flagged = 0
                for nr, nc in get_neighbors(r, c, game.squares_y, game.squares_x):
                    if game.grid[nr][nc].has_flag:
                        flagged += 1
                    elif (nr, nc) in frontier:
                        adj_frontier.append((nr, nc))
                required = cell.bomb_count - flagged
                if adj_frontier and 0 <= required <= len(adj_frontier):
                    constraints[(r, c)] = (required, adj_frontier)
    return constraints

def group_frontier_by_constraints(frontier, constraints):
    """
    Build an undirected graph among frontier cells that share constraints,
    then find connected components (clusters).
    """
    graph = {cell: set() for cell in frontier}
    for _, (req, cells) in constraints.items():
        for cellA in cells:
            for cellB in cells:
                if cellA != cellB:
                    graph[cellA].add(cellB)
                    graph[cellB].add(cellA)

    clusters = []
    visited = set()
    for cell in frontier:
        if cell not in visited:
            stack = [cell]
            comp = []
            while stack:
                cur = stack.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                comp.append(cur)
                for nxt in graph[cur]:
                    if nxt not in visited:
                        stack.append(nxt)
            clusters.append(comp)
    return clusters

def get_cluster_constraints(cluster, constraints):
    """
    Extract constraints relevant only to the cells in 'cluster'
    """
    cluster_set = set(cluster)
    cluster_constraints = {}
    for clue, (req, frontier_cells) in constraints.items():
        intersected = [fc for fc in frontier_cells if fc in cluster_set]
        if intersected:
            cluster_constraints[clue] = (req, intersected)
    return cluster_constraints

def valid_partial(assignment, constraints_list):
    """
    partial assignment: array of -1 (unassigned), 0 (no bomb), 1 (bomb)
    constraints_list: list of (required_bombs, [indices in cluster])
    Check if partial assignment so far doesn't violate any constraint.
    """
    for req, indices in constraints_list:
        assigned_sum = 0
        unassigned = 0
        for idx in indices:
            val = assignment[idx]
            if val == -1:
                unassigned += 1
            else:
                assigned_sum += val
        # If we exceed required bombs, or can't possibly meet it => invalid
        if assigned_sum > req:
            return False
        if assigned_sum + unassigned < req:
            return False
    return True

def backtrack_csp(i, assignment, constraints_list, results, n):
    if i == n:
        # Check final validity
        for req, indices in constraints_list:
            if sum(assignment[idx] for idx in indices) != req:
                return
        # Valid full assignment
        results['count'] += 1
        for j in range(n):
            if assignment[j] == 1:
                results['bomb_counts'][j] += 1
        return

    for val in [0, 1]:
        assignment[i] = val
        if valid_partial(assignment, constraints_list):
            backtrack_csp(i+1, assignment, constraints_list, results, n)
    assignment[i] = -1  # revert

def csp_cluster_solver(cluster, cluster_constraints):
    """
    Returns { cell: probability_of_bomb }
    """
    n = len(cluster)
    index_map = {cell: i for i, cell in enumerate(cluster)}

    # Build constraint list
    constraints_list = []
    for clue, (req, frontier_cells) in cluster_constraints.items():
        indices = [index_map[cell] for cell in frontier_cells]
        constraints_list.append((req, indices))

    # Prepare structure to gather solutions
    results = {
        'count': 0,
        'bomb_counts': [0]*n
    }
    assignment = [-1]*n

    backtrack_csp(0, assignment, constraints_list, results, n)

    if results['count'] == 0:
        # No valid solutions => uncertain => assume 1.0
        return {cell: 1.0 for cell in cluster}
    else:
        probs = {}
        for i, cell in enumerate(cluster):
            probs[cell] = results['bomb_counts'][i] / results['count']
        return probs


# -------------------- "Improved" CSP Solver --------------------
def csp_solver(game):
    """
    1) If the game isn't initialized, pick a RANDOM hidden cell.
    2) Otherwise:
       - Build constraints from revealed clues
       - Solve each connected cluster in the frontier
       - If there's any cell with probability=0.0 => pick from them randomly
       - Otherwise pick random from all hidden/unflagged
    """
    hidden_cells = [
        (r, c) for r in range(game.squares_y) for c in range(game.squares_x)
        if (not game.grid[r][c].is_visible and not game.grid[r][c].has_flag)
    ]

    # If board not init, just pick a random hidden cell
    if not game.init:
        if hidden_cells:
            return random.choice(hidden_cells)
        else:
            return None

    # Build constraints
    frontier = get_frontier_cells(game)
    constraints = get_constraints(game, frontier)

    # If frontier is empty (and game not won/lost), pick random from hidden
    if not frontier:
        if hidden_cells:
            return random.choice(hidden_cells)
        return None

    clusters = group_frontier_by_constraints(frontier, constraints)
    probabilities = {}

    # Solve each cluster
    for cluster in clusters:
        cluster_constraints = get_cluster_constraints(cluster, constraints)
        cluster_probs = csp_cluster_solver(cluster, cluster_constraints)
        probabilities.update(cluster_probs)

    # For hidden cells not in frontier => default uniform probability
    flagged_count = sum(
        1 for r in range(game.squares_y) for c in range(game.squares_x)
        if game.grid[r][c].has_flag
    )
    total_unrevealed = len(hidden_cells)
    default_prob = 1.0
    if total_unrevealed > 0:
        bombs_left = game.num_bombs - flagged_count
        default_prob = bombs_left / total_unrevealed

    for (r, c) in hidden_cells:
        if (r, c) not in probabilities:
            probabilities[(r, c)] = default_prob

    # Now see if there's any guaranteed safe cell (prob=0.0)
    guaranteed_safe = [cell for cell, prob in probabilities.items() if prob == 0.0]
    if guaranteed_safe:
        return random.choice(guaranteed_safe)

    # Otherwise pick a random hidden cell
    if hidden_cells:
        return random.choice(hidden_cells)

    return None


# -------------------- Pygame Initialization and Main Loop --------------------
pygame.init()
size = (NSQUARES_X * (WIDTH + MARGIN) + MARGIN,
        (NSQUARES_Y * (HEIGHT + MARGIN) + MARGIN) + MENU_SIZE)
screen = pygame.display.set_mode(size, pygame.RESIZABLE)
pygame.display.set_caption("Minesweeper CSP Solver (Improved)")
font = pygame.font.Font('freesansbold.ttf', 24)
game = Game()
menu = Menu()
clock = pygame.time.Clock()

# Auto-solver variables
auto_solve = True
last_auto_move_time = 0
auto_move_delay = 500  # milliseconds between auto moves

# Move history (for printing steps after game ends)
move_history = []
steps_printed = False  # ensure we only print once

def run_game():
    global auto_solve, last_auto_move_time, steps_printed
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                position = pygame.mouse.get_pos()
                column = position[0] // (WIDTH + MARGIN)
                row = (position[1] - MENU_SIZE) // (HEIGHT + MARGIN)
                if row >= game.squares_y:
                    row = game.squares_y - 1
                if column >= game.squares_x:
                    column = game.squares_x - 1
                if row >= 0:
                    game.click_handle(row, column, event.button)
                else:
                    menu.click_handle(game)
            elif event.type == pygame.VIDEORESIZE:
                if game.resize:
                    game.adjust_grid(event.w, event.h)
                    game.reset_game()
                    move_history.clear()
                    steps_printed = False
                else:
                    game.resize = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    auto_solve = not auto_solve
                    print("Auto-solver toggled:", auto_solve)
                if event.key == pygame.K_r:
                    game.reset_game()
                    move_history.clear()
                    steps_printed = False
        
        # Auto-solver logic
        current_time = pygame.time.get_ticks()
        if (auto_solve and not game.game_lost and not game.game_won
            and current_time - last_auto_move_time > auto_move_delay):
            best_move = csp_solver(game)
            if best_move is not None:
                r, c = best_move
                move_history.append((r, c))
                game.click_handle(r, c, LEFT_CLICK)
                last_auto_move_time = current_time

        # If the game ends, print move history if not done already
        if (game.game_lost or game.game_won) and not steps_printed:
            print("Game ended.")
            print("Moves played by the solver:")
            for idx, (r, c) in enumerate(move_history):
                print(f"  Step {idx+1}: Clicked cell ({r}, {c})")
            steps_printed = True

        game.draw()
        menu.draw(game)
        clock.tick(60)
        pygame.display.flip()

if __name__ == "__main__":
    run_game()
