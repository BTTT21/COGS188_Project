"""
NNsolver.py

A sample neural network solver for Minesweeper.
This version defines a simple convolutional neural network (using PyTorch) that is intended to
predict the "safety" (i.e. the probability of being free of a mine) for each cell on the board.
It converts the Minesweeper board (from minesweeper.py) into a 3‐channel tensor and, on each move,
selects the covered, unflagged cell with the highest predicted safety.

NOTE:
  - This sample network is not trained (unless you provide trained weights). In practice, you would
    need to train the network using a large dataset or via reinforcement learning.
  - The board is represented by three channels:
      Channel 0: Visible indicator (1 if cell is visible, else 0).
      Channel 1: Bomb count (normalized by 8) for visible cells; 0 for hidden.
      Channel 2: Flag indicator (1 if flagged, else 0).
  - The network outputs a safety score (values between 0 and 1) for each cell.
  
Usage:
    python NNsolver.py
This will open the Minesweeper window (using your existing minesweeper.py) and automatically make moves
based on the neural network’s prediction.
"""

import pygame
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from random import choice

# Import the game environment and constants from your existing minesweeper.py
from minesweeper import Game, Menu
from minesweeper import BLACK, WHITE, BLUE, RED, GRAY, MARGIN, WIDTH, HEIGHT, MENU_SIZE, LEFT_CLICK, RIGHT_CLICK

###############################################################################
# Neural Network Model Definition
###############################################################################

class MinesweeperNN(nn.Module):
    def __init__(self, input_channels=3):
        super(MinesweeperNN, self).__init__()
        # A simple convolutional network
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        
    def forward(self, x):
        # Apply ReLU after first two conv layers, and finally a sigmoid to produce probabilities
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = torch.sigmoid(x)  # values between 0 and 1: higher means safer
        return x

###############################################################################
# Neural Network Based Move Selector
###############################################################################

class NNMoveSelector:
    def __init__(self, model=None, device=None):
        """
        :param model: A pretrained MinesweeperNN model. If None, a new untrained model is created.
        :param device: Torch device (e.g. torch.device("cpu") or "cuda")
        """
        self.device = device if device is not None else torch.device("cpu")
        if model is None:
            self.model = MinesweeperNN()
            # To use a trained model, load its state here:
            # self.model.load_state_dict(torch.load("minesweeper_model.pt", map_location=self.device))
        else:
            self.model = model
        self.model.to(self.device)
        self.model.eval()

    def board_to_tensor(self, game):
        """
        Convert the game board to a tensor with shape (1, 3, rows, cols).
        Representation:
          - Channel 0: 1 if cell is visible, else 0.
          - Channel 1: For visible cells, normalized bomb count (bomb_count/8); 0 otherwise.
          - Channel 2: 1 if cell is flagged, else 0.
        """
        rows = game.squares_y
        cols = game.squares_x
        visible = np.zeros((rows, cols), dtype=np.float32)
        bomb_count = np.zeros((rows, cols), dtype=np.float32)
        flagged = np.zeros((rows, cols), dtype=np.float32)
        for i in range(rows):
            for j in range(cols):
                cell = game.grid[i][j]
                if cell.is_visible:
                    visible[i, j] = 1.0
                    bomb_count[i, j] = cell.bomb_count / 8.0  # normalize
                if cell.has_flag:
                    flagged[i, j] = 1.0
        board = np.stack([visible, bomb_count, flagged], axis=0)
        board = np.expand_dims(board, axis=0)  # add batch dimension
        board_tensor = torch.from_numpy(board)
        return board_tensor.to(self.device)

    def choose_move(self, game):
        """
        Use the neural network to predict a safety score for each cell and choose the cell
        with the highest safety score among those that are not visible and not flagged.
        """
        board_tensor = self.board_to_tensor(game)
        with torch.no_grad():
            output = self.model(board_tensor)  # shape: (1, 1, rows, cols)
        prob_map = output.squeeze().cpu().numpy()  # shape: (rows, cols)
        candidate_moves = []
        candidate_scores = []
        for i in range(game.squares_y):
            for j in range(game.squares_x):
                cell = game.grid[i][j]
                if not cell.is_visible and not cell.has_flag:
                    candidate_moves.append((i, j))
                    candidate_scores.append(prob_map[i, j])
        if not candidate_moves:
            return None
        # Choose the move with the highest predicted safety (score)
        best_idx = candidate_scores.index(max(candidate_scores))
        return candidate_moves[best_idx]

    def solve(self, game):
        """
        Return the coordinate (row, col) of the move chosen by the neural network.
        """
        return self.choose_move(game)

###############################################################################
# Main Loop: Automatic Neural Network Based Minesweeper Solver
###############################################################################

if __name__ == "__main__":
    pygame.init()
    game = Game()
    menu = Menu()
    clock = pygame.time.Clock()
    auto_solve = True  # Enable automatic moves

    # Create an NN move selector.
    # For a trained model, load weights; here we use an untrained model.
    nn_solver = NNMoveSelector()

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

        # Automatic move: if game is ongoing, let the NN select a move.
        if auto_solve and not game.game_lost and not game.game_won:
            move = nn_solver.solve(game)
            if move is not None:
                r, c = move
                if not game.grid[r][c].is_visible:
                    game.click_handle(r, c, LEFT_CLICK)
                    pygame.time.delay(500)  # Delay for visualization

        game.draw()
        menu.draw(game)
        clock.tick(10)
        pygame.display.flip()