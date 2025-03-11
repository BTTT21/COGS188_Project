import pygame
import sys
import random
from random import randrange

# Colors used
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
GRAY = (127, 127, 127)

# Sets the WIDTH and HEIGHT of each grid cell
WIDTH = 30
HEIGHT = 30
MARGIN = 5
MENU_SIZE = 40
LEFT_CLICK = 1
RIGHT_CLICK = 3

# Expert board settings
NSQUARES_X = 16  # Number of columns
NSQUARES_Y = 16  # Number of rows
# Set initial number of bombs to 40 for Expert level
EXPERT_BOMBS = 40

# Class that holds the game logic          
class Game:
    def __init__(self, use_display=True, num_bombs=EXPERT_BOMBS, fixed_seed=None):
        self.use_display = use_display
        self.squares_x = NSQUARES_X
        self.squares_y = NSQUARES_Y
        self.num_bombs = num_bombs
        self.fixed_seed = fixed_seed
        self.grid = [[self.Cell(x, y) for x in range(self.squares_x)] for y in range(self.squares_y)]
        self.init = False
        self.game_lost = False
        self.game_won = False
        self.resize = False
        self.flag_count = 0
        
        if self.use_display:
            pygame.init()
            global screen, font
            size = (NSQUARES_X * (WIDTH + MARGIN) + MARGIN, (NSQUARES_Y * (HEIGHT + MARGIN) + MARGIN) + MENU_SIZE)
            screen = pygame.display.set_mode(size, pygame.RESIZABLE)
            pygame.display.set_caption("Minesweeper by Raul Vieira - Expert Level")
            font = pygame.font.Font('freesansbold.ttf', 24)

    def draw(self):
        # Set the screen background color
        screen.fill(BLACK)
        # Draw the grid
        for row in range(self.squares_y):
            for column in range(self.squares_x):
                color = WHITE
                if self.grid[row][column].is_visible:
                    color = RED if self.grid[row][column].has_bomb else GRAY  
                elif self.grid[row][column].has_flag:
                    color = BLUE

                if self.use_display:
                    pygame.draw.rect(screen,
                                color,
                                [(MARGIN + WIDTH) * column + MARGIN,
                                (MARGIN + HEIGHT) * row + MARGIN + MENU_SIZE,
                                WIDTH,
                                HEIGHT])
                    self.grid[row][column].show_text()
        
    # Adjusts the grid when the screen size has changed
    def adjust_grid(self, sizex, sizey):
        global screen
        self.squares_x = (sizex - MARGIN) // (WIDTH + MARGIN)
        self.squares_y = (sizey - MARGIN - MENU_SIZE) // (HEIGHT + MARGIN)
        if self.squares_x < 8:
            self.squares_x = 8
        if self.squares_y < 8:
            self.squares_y = 8
        if self.num_bombs > (self.squares_x * self.squares_y) // 3:
            self.num_bombs = self.squares_x * self.squares_y // 3
        self.grid = [[self.Cell(x, y) for x in range(self.squares_x)] for y in range(self.squares_y)]
        size = ((self.squares_x*(WIDTH + MARGIN) + MARGIN), (self.squares_y*(HEIGHT + MARGIN) + MARGIN + MENU_SIZE))
        screen = pygame.display.set_mode(size, pygame.RESIZABLE)

    # Reveals all bomb cells when user loses
    def game_over(self):
        for row in range(self.squares_y):
            for column in range(self.squares_x):
                if self.grid[row][column].has_bomb:
                    self.grid[row][column].is_visible = True
                self.grid[row][column].has_flag = False

    # Changes the number of bombs placed and caps it
    def change_num_bombs(self, bombs):
        self.num_bombs += bombs
        if self.num_bombs < 1:
            self.num_bombs = 1
        elif self.num_bombs > (self.squares_x * self.squares_y) // 3:
            self.num_bombs = self.squares_x * self.squares_y // 3
        self.reset_game() 

    
    # 专门生成固定炸弹，不依赖点击
    def generate_fixed_bombs(self, seed):
        random.seed(seed)
        for row in range(self.squares_y):
            for column in range(self.squares_x):
                self.grid[row][column].has_bomb = False
        bombplaced = 0
        while bombplaced < self.num_bombs:
            x = randrange(self.squares_y)
            y = randrange(self.squares_x)
            if not self.grid[x][y].has_bomb:
                self.grid[x][y].has_bomb = True
                bombplaced += 1
        self.count_all_bombs()
    
    # Place bombs randomly (ensuring the first click is safe)
    # 添加 seed 参数，固定雷区
    def place_bombs(self, row, column, seed=None):
        if seed is not None:
            random.seed(seed)  # 固定种子
        bombplaced = 0
        while bombplaced < self.num_bombs:
            x = randrange(self.squares_y)
            y = randrange(self.squares_x)
            # 确保第一次点击不会踩雷
            if not self.grid[x][y].has_bomb and (x, y) != (row, column):
                self.grid[x][y].has_bomb = True
                bombplaced += 1
        self.count_all_bombs()


        
    # Count bombs adjacent to every cell in the grid
    def count_all_bombs(self):
        for row in range(self.squares_y):
            for column in range(self.squares_x):
                self.grid[row][column].count_bombs(self.squares_y, self.squares_x, self.grid)
    
    def reset_game(self, keep_bombs=False):
        for row in range(self.squares_y):
            for column in range(self.squares_x):
                self.init = False
                self.grid[row][column].is_visible = False
                self.grid[row][column].bomb_count = 0
                self.grid[row][column].test = False
                self.grid[row][column].has_flag = False
                self.game_lost = False
                self.game_won = False
                self.flag_count = 0
                if not keep_bombs:
                    self.grid[row][column].has_bomb = False

        # 如果有固定 seed 且不保留炸弹，则生成固定炸弹
        if not keep_bombs and self.fixed_seed is not None:
            self.generate_fixed_bombs(self.fixed_seed)



    def check_victory(self):   
        count = 0
        total = self.squares_x * self.squares_y
        for row in range(self.squares_y):
            for column in range(self.squares_x):
                if self.grid[row][column].is_visible:
                    count += 1
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

    # Handle for grid clicks
    def click_handle(self, row, column, button):
        if button == LEFT_CLICK and self.game_won:
            self.reset_game()
        elif button == LEFT_CLICK and not self.grid[row][column].has_flag: 
            if not self.game_lost:
                # Place bombs on the first click so you never click a bomb first
                if not self.init:
                    self.place_bombs(row, column)
                    self.init = True
                # Reveal the clicked cell
                self.grid[row][column].is_visible = True
                self.grid[row][column].has_flag = False
                if self.grid[row][column].has_bomb:
                    self.game_over()
                    self.game_lost = True
                if self.grid[row][column].bomb_count == 0 and not self.grid[row][column].has_bomb:
                    self.grid[row][column].open_neighbours(self.squares_y, self.squares_x, self.grid)
                self.check_victory()
            else:
                self.game_lost = False
                self.reset_game()
        
        elif button == RIGHT_CLICK and not self.game_won:
            if not self.grid[row][column].has_flag:
                if self.flag_count < self.num_bombs and not self.grid[row][column].is_visible:
                    self.grid[row][column].has_flag = True
            else:
                self.grid[row][column].has_flag = False
            self.count_flags()


    # Game Sub-Class for each cell of the grid
    class Cell:
        def __init__(self, x, y):
            self.x = x  # column index
            self.y = y  # row index
            self.is_visible = False
            self.has_bomb = False
            self.bomb_count = 0
            self.text = ""
            self.test = False
            self.has_flag = False

        # Display the bomb count text for the cell
        def show_text(self):
            if self.is_visible:
                if self.bomb_count == 0:
                    self.text = font.render("", True, BLACK)
                else:
                    self.text = font.render(str(self.bomb_count), True, BLACK)
                screen.blit(self.text, (self.x * (WIDTH + MARGIN) + 12, self.y * (HEIGHT + MARGIN) + 10 + MENU_SIZE))
        
        # Count how many bombs are next to this cell (3x3)
        def count_bombs(self, max_rows, max_cols, grid):
            if not self.test:
                self.test = True
                if not self.has_bomb:
                    for col in range(self.x - 1, self.x + 2):
                        for row in range(self.y - 1, self.y + 2):
                            if (row >= 0 and row < max_rows and 
                                col >= 0 and col < max_cols and 
                                not (col == self.x and row == self.y) and 
                                grid[row][col].has_bomb):  # 用传进来的 grid
                                self.bomb_count += 1

        
        # Open all neighboring cells if this cell has zero adjacent bombs
        def open_neighbours(self, max_rows, max_cols, grid):
            col = self.x
            row = self.y
            for row_off in range(-1, 2):
                for col_off in range(-1, 2):
                    # Check only vertical and horizontal neighbours
                    if ((row_off == 0 or col_off == 0) and row_off != col_off and
                        row + row_off >= 0 and col + col_off >= 0 and 
                        row + row_off < max_rows and col + col_off < max_cols):
                
                        grid[row + row_off][col + col_off].count_bombs(max_rows, max_cols, grid)
                
                        if (not grid[row + row_off][col + col_off].is_visible and 
                            not grid[row + row_off][col + col_off].has_bomb):  
                            grid[row + row_off][col + col_off].is_visible = True
                            grid[row + row_off][col + col_off].has_flag = False
                            if grid[row + row_off][col + col_off].bomb_count == 0: 
                                grid[row + row_off][col + col_off].open_neighbours(max_rows, max_cols, grid)


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
        self.label_bombs.show(screen, obj.num_bombs)
        self.label_flags.show(screen, obj.flag_count)
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
            if pos[0] > self.x and pos[1] > self.y and pos[0] < (self.x + self.width) and pos[1] < (self.y + self.height):
                return True
            else:
                return False



# Main loop

def run_game():
    pygame.init()
    size = (NSQUARES_X * (WIDTH + MARGIN) + MARGIN, (NSQUARES_Y * (HEIGHT + MARGIN) + MARGIN) + MENU_SIZE)
    screen = pygame.display.set_mode(size, pygame.RESIZABLE)
    pygame.display.set_caption("Minesweeper by Raul Vieira - Expert Level")
    font = pygame.font.Font('freesansbold.ttf', 24)
    game = Game()  # 这里传 True 或不传，默认显示
    menu = Menu()
    clock = pygame.time.Clock()
    
    while True:
        for event in pygame.event.get():
            # Closes the game if user clicks the X
            if event.type == pygame.QUIT:  
                pygame.quit()
                sys.exit()
            # Handle mouse clicks
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
            # Handle screen resize events
            elif event.type == pygame.VIDEORESIZE:
                if game.resize: 
                    game.adjust_grid(event.w, event.h)
                    game.reset_game()
                else:  
                    game.resize = True
    
        game.draw()
        menu.draw(game)
        clock.tick(40)
        pygame.display.flip()


if __name__ == "__main__":
    run_game()