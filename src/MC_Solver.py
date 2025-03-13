import random
import numpy as np
import matplotlib.pyplot as plt
import time
from minesweeper_MC import Game, LEFT_CLICK

class MonteCarloSolver:
    def __init__(self, game, episodes=2000, gamma=0.95):
        self.game = game
        self.episodes = episodes
        self.gamma = gamma
        self.Q = {}  # (state, action) -> value
        self.returns_sum = {}  # (state, action) -> total return
        self.returns_count = {}  # (state, action) -> count
        self.policy = {}  # state -> action
        self.train_results = []  # Track win/loss results
        
        # Parameters for adaptive exploration
        self.epsilon_start = 0.9
        self.epsilon_end = 0.1
        
        # Track training progress
        self.episode_lengths = []
        self.episode_rewards = []

    def observe_state(self):
        """
        Simplified state representation focusing on visible patterns.
        We represent each cell as:
        -1: Hidden
        0-8: Visible with that many bombs nearby
        """
        visible_grid = []
        for r in range(self.game.squares_y):
            row = []
            for c in range(self.game.squares_x):
                cell = self.game.grid[r][c]
                if cell.is_visible:
                    row.append(cell.bomb_count)
                else:
                    row.append(-1)
            visible_grid.append(tuple(row))
        return tuple(visible_grid)

    def get_local_state(self, row, col, radius=1):
        """
        Get a local view of the state centered at (row, col) with given radius.
        This creates a smaller, more generalizable state representation.
        """
        local_state = []
        for r in range(row - radius, row + radius + 1):
            local_row = []
            for c in range(col - radius, col + radius + 1):
                if 0 <= r < self.game.squares_y and 0 <= c < self.game.squares_x:
                    cell = self.game.grid[r][c]
                    if cell.is_visible:
                        local_row.append(cell.bomb_count)
                    else:
                        local_row.append(-1)
                else:
                    # Out of bounds
                    local_row.append(-2)
            local_state.append(tuple(local_row))
        return tuple(local_state)

    def get_unknown_cells(self):
        """Return all hidden cells"""
        return [(r, c) for r in range(self.game.squares_y) for c in range(self.game.squares_x)
                if not self.game.grid[r][c].is_visible]

    def get_border_cells(self):
        """
        Get cells that are hidden but adjacent to at least one revealed cell.
        These are the most informative cells to click.
        """
        border_cells = []
        for r in range(self.game.squares_y):
            for c in range(self.game.squares_x):
                if self.game.grid[r][c].is_visible:
                    continue
                    
                # Check if this hidden cell is adjacent to any revealed cell
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < self.game.squares_y and 
                            0 <= nc < self.game.squares_x and 
                            self.game.grid[nr][nc].is_visible):
                            border_cells.append((r, c))
                            break
                    else:
                        continue
                    break
        
        return border_cells if border_cells else self.get_unknown_cells()

    def safe_cells_from_logic(self):
        """Identify cells that are definitely safe based on basic Minesweeper logic"""
        safe_cells = []
        
        for r in range(self.game.squares_y):
            for c in range(self.game.squares_x):
                cell = self.game.grid[r][c]
                if not cell.is_visible or cell.bomb_count == 0:
                    continue
                    
                # Get hidden and flagged neighbors
                hidden_neighbors = []
                
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < self.game.squares_y and 
                            0 <= nc < self.game.squares_x and 
                            not self.game.grid[nr][nc].is_visible):
                            hidden_neighbors.append((nr, nc))
                
                # If the number of hidden neighbors equals the bomb count, all other hidden 
                # neighbors are safe (this is a simplification as we're not tracking flags)
                if len(hidden_neighbors) == cell.bomb_count:
                    # All hidden neighbors potentially contain bombs
                    continue
                    
                if len(hidden_neighbors) > cell.bomb_count:
                    # Some hidden neighbors must be safe
                    # For simplicity, we'll just add them all as candidates
                    safe_cells.extend(hidden_neighbors)
        
        # Remove duplicates
        return list(set(safe_cells))

    def get_epsilon(self, episode):
        """Calculate adaptive exploration rate"""
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * (
            1 - min(1.0, episode / (self.episodes * 0.7))
        )

    def behavior_policy(self, episode_num):
        """
        Policy for generating episodes:
        1. Start with corners or edges (safer in Minesweeper)
        2. Use logical deduction when possible
        3. Otherwise use an epsilon-greedy approach with localized states
        """
        epsilon = self.get_epsilon(episode_num)
        
        # Get current state
        state = self.observe_state()
        
        # First move strategy - corners and edges are statistically safer
        if all(all(cell == -1 for cell in row) for row in state):
            corners = [(0, 0), (0, self.game.squares_x-1), 
                      (self.game.squares_y-1, 0), (self.game.squares_y-1, self.game.squares_x-1)]
            for corner in corners:
                if not self.game.grid[corner[0]][corner[1]].is_visible:
                    return corner
            # If no corners available, try an edge
            edges = []
            for r in range(self.game.squares_y):
                edges.extend([(r, 0), (r, self.game.squares_x-1)])
            for c in range(self.game.squares_x):
                edges.extend([(0, c), (self.game.squares_y-1, c)])
            random.shuffle(edges)
            for edge in edges:
                if not self.game.grid[edge[0]][edge[1]].is_visible:
                    return edge
        
        # Check for logically safe cells
        safe_cells = self.safe_cells_from_logic()
        if safe_cells:
            return random.choice(safe_cells)
        
        # Epsilon-greedy approach with border cells
        border_cells = self.get_border_cells()
        if not border_cells:
            return None
        
        if random.random() < epsilon:
            # Exploration - random action, but prefer border cells
            return random.choice(border_cells)
        else:
            # Exploitation - use Q values with localized states
            best_action = None
            best_value = float('-inf')
            
            # Try each border cell and check its Q value with local state
            for r, c in border_cells:
                local_state = self.get_local_state(r, c)
                if (local_state, (r, c)) in self.Q:
                    q_value = self.Q[(local_state, (r, c))]
                    if q_value > best_value:
                        best_value = q_value
                        best_action = (r, c)
            
            # If we found a good action based on Q values, use it
            if best_action is not None:
                return best_action
            
            # Otherwise, pick a random border cell
            return random.choice(border_cells)

    def click_cell(self, row, col):
        """Execute a click on the specified cell"""
        self.game.click_handle(row, col, LEFT_CLICK)
        self.game.check_victory()

    def generate_episode(self, episode_num, max_steps=100):
        """Generate a complete Minesweeper episode"""
        self.game.reset_game(keep_bombs=False)
        episode = []
        visited_states_actions = set()
        states_actions_history = []  # Track the history for calculating return
        
        step = 0
        episode_reward = 0
        
        while not self.game.game_lost and not self.game.game_won and step < max_steps:
            action = self.behavior_policy(episode_num)
            if action is None:
                break
                
            # Skip revisited state-actions to avoid loops
            local_state = self.get_local_state(action[0], action[1])
            if (local_state, action) in visited_states_actions:
                break
                
            # Track previously visible cells for reward calculation
            visible_before = sum(1 for row in self.game.grid for cell in row if cell.is_visible)
            
            # Execute action
            self.click_cell(*action)
            
            # Calculate reward - this is critical for effective learning
            visible_after = sum(1 for row in self.game.grid for cell in row if cell.is_visible)
            new_cells_revealed = visible_after - visible_before
            
            # Reward structure:
            # 1. +5 for winning
            # 2. -5 for losing
            # 3. +0.5 per newly revealed cell (progress reward)
            # 4. -0.1 per step (efficiency incentive)
            if self.game.game_won:
                reward = 5 
            elif self.game.game_lost:
                reward = -5
            else:
                reward = 0.5 * new_cells_revealed - 0.1
                
            # Add to episode history
            states_actions_history.append((local_state, action, reward))
            visited_states_actions.add((local_state, action))
            episode_reward += reward
            step += 1
            
            # For first-visit MC, only include first visits to each state-action pair
            if (local_state, action) not in [(s, a) for s, a, _ in episode]:
                episode.append((local_state, action, reward))
        
        self.episode_lengths.append(step)
        self.episode_rewards.append(episode_reward)
        
        # Record the outcome (win or loss)
        self.train_results.append(1 if self.game.game_won else 0)
        
        return episode, states_actions_history

    def update_q_values(self, episode_history):
        """Update Q values using first-visit Monte Carlo with discounting"""
        G = 0  # Return (cumulative discounted reward)
        
        # Process the episode in reverse order
        for t in range(len(episode_history) - 1, -1, -1):
            state, action, reward = episode_history[t]
            G = self.gamma * G + reward
            
            # First-visit update for Q values
            sa_pair = (state, action)
            
            # Incremental update formula
            self.returns_count[sa_pair] = self.returns_count.get(sa_pair, 0) + 1
            learning_rate = 1.0 / self.returns_count[sa_pair]  # Decreasing learning rate
            current_estimate = self.Q.get(sa_pair, 0)
            
            # Update Q value with weighted average
            self.Q[sa_pair] = current_estimate + learning_rate * (G - current_estimate)

    def extract_policy(self):
        """Extract a deterministic policy from the Q values"""
        self.policy = {}
        
        # Group Q values by state
        state_actions = {}
        for (state, action), value in self.Q.items():
            if state not in state_actions:
                state_actions[state] = []
            state_actions[state].append((action, value))
        
        # For each state, select the action with highest Q value
        for state, actions in state_actions.items():
            if actions:
                best_action = max(actions, key=lambda x: x[1])[0]
                self.policy[state] = best_action

    def train(self, verbose=True):
        """Train the agent using Monte Carlo method"""
        if verbose:
            print("Starting training...")
        
        # Track win rates for plotting
        window_size = 100
        win_rates = []
        
        for ep in range(1, self.episodes + 1):
            # Generate episode
            _, episode_history = self.generate_episode(ep)
            
            # Update Q values
            self.update_q_values(episode_history)
            
            # Update win rate tracking (every window_size episodes)
            if ep % window_size == 0:
                recent_win_rate = sum(self.train_results[-window_size:]) / window_size
                win_rates.append(recent_win_rate)
                
                if verbose:
                    print(f"Episode {ep}/{self.episodes} - Recent win rate: {recent_win_rate:.2f}")
                    print(f"Q table size: {len(self.Q)}")
                    print(f"Average episode length: {sum(self.episode_lengths[-window_size:]) / window_size:.1f}")
        
        # Extract policy from final Q values
        self.extract_policy()
        
        if verbose:
            print("Training completed!")
            print(f"Final Q table size: {len(self.Q)}")
            print(f"Final policy size: {len(self.policy)}")
        
        return win_rates

    def play_game(self, use_policy=True, max_steps=100):
        """Play a single game using the learned policy or behavior policy"""
        self.game.reset_game(keep_bombs=False)
        steps = 0
        
        while not self.game.game_lost and not self.game.game_won and steps < max_steps:
            if use_policy:
                # Use learned policy where available
                found_action = False
                unknown_cells = self.get_unknown_cells()
                
                if not unknown_cells:
                    break
                
                # Try all unknown cells and see if we have a policy for any of them
                for r, c in unknown_cells:
                    local_state = self.get_local_state(r, c)
                    if local_state in self.policy:
                        action = self.policy[local_state]
                        found_action = True
                        break
                
                # If no policy found, use a safe cell or random border cell
                if not found_action:
                    safe_cells = self.safe_cells_from_logic()
                    if safe_cells:
                        action = random.choice(safe_cells)
                    else:
                        border_cells = self.get_border_cells()
                        action = random.choice(border_cells) if border_cells else random.choice(unknown_cells)
            else:
                # Use behavior policy
                action = self.behavior_policy(self.episodes)  # Use final epsilon
            
            if action is None:
                break
                
            self.click_cell(*action)
            steps += 1
        
        return self.game.game_won, steps

    def evaluate(self, num_games=100, use_policy=True):
        """Evaluate the agent's performance"""
        wins = 0
        total_steps = 0
        
        for _ in range(num_games):
            win, steps = self.play_game(use_policy=use_policy)
            if win:
                wins += 1
                total_steps += steps
        
        win_rate = wins / num_games
        avg_steps = total_steps / wins if wins > 0 else 0
        
        print(f"Evaluation over {num_games} games:")
        print(f"Win rate: {win_rate:.2f}")
        print(f"Average steps to win: {avg_steps:.1f}")
        
        return win_rate, avg_steps

    def test_win_rate(self, num_games=100, verbose=True):
        """Test win rate over a number of games with progress reporting"""
        wins = 0
        losses = 0
        total_exploration = 0
        start_time = time.time()
        
        for i in range(1, num_games + 1):
            # Reset the game with a new bomb placement
            self.game.reset_game(keep_bombs=False)
            game_over = False
            steps = 0
            
            while not game_over and steps < 100:  # Limit steps to prevent infinite loops
                # Get action using the trained policy/behavior
                action = self.behavior_policy(self.episodes)  # Use behavior policy
                
                if action is None:
                    break
                
                # Execute action
                self.click_cell(*action)
                steps += 1
                
                # Check if game is over
                if self.game.game_won:
                    wins += 1
                    game_over = True
                elif self.game.game_lost:
                    losses += 1
                    game_over = True
            
            # Add exploration rate for this game
            total_exploration += 1 if self.game.game_won else 0
            
            if verbose and i % 10 == 0:
                print(f"Progress: {i}/{num_games} games played")
        
        # Calculate statistics
        win_rate = (wins / num_games) * 100
        avg_exploration = (total_exploration / num_games) * 100
        time_taken = time.time() - start_time
        
        if verbose:
            print("----- RESULTS -----")
            print(f"Games played: {num_games}")
            print(f"Number of mines: {self.game.num_bombs}")
            print(f"Grid size: {self.game.squares_x}x{self.game.squares_y}")
            print(f"Wins: {wins}")
            print(f"Losses: {losses}")
            print(f"Win rate: {win_rate:.2f}%")
            print(f"Average Exploration Rate: {avg_exploration:.2f}%")
            print(f"Time taken: {time_taken:.2f} seconds")
        
        return win_rate

    def plot_training_progress(self):
        """Plot training progress metrics"""
        plt.figure(figsize=(15, 10))
        
        # Plot win rate
        plt.subplot(2, 2, 1)
        window_size = 100
        win_rates = []
        for i in range(window_size, len(self.train_results) + 1, window_size):
            win_rates.append(sum(self.train_results[i-window_size:i]) / window_size)
        
        plt.plot(range(window_size, len(self.train_results) + 1, window_size), win_rates, 'b-o')
        plt.title('Win Rate (per 100 episodes)')
        plt.xlabel('Episodes')
        plt.ylabel('Win Rate')
        plt.grid(True)
        
        # Plot episode lengths
        plt.subplot(2, 2, 2)
        avg_lengths = []
        for i in range(window_size, len(self.episode_lengths) + 1, window_size):
            avg_lengths.append(sum(self.episode_lengths[i-window_size:i]) / window_size)
        
        plt.plot(range(window_size, len(self.episode_lengths) + 1, window_size), avg_lengths, 'g-o')
        plt.title('Average Episode Length (per 100 episodes)')
        plt.xlabel('Episodes')
        plt.ylabel('Steps')
        plt.grid(True)
        
        # Plot episode rewards
        plt.subplot(2, 2, 3)
        avg_rewards = []
        for i in range(window_size, len(self.episode_rewards) + 1, window_size):
            avg_rewards.append(sum(self.episode_rewards[i-window_size:i]) / window_size)
        
        plt.plot(range(window_size, len(self.episode_rewards) + 1, window_size), avg_rewards, 'r-o')
        plt.title('Average Episode Reward (per 100 episodes)')
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.grid(True)
        
        # Plot Q-table size growth
        plt.subplot(2, 2, 4)
        plt.hist(list(self.Q.values()), bins=20)
        plt.title(f'Q-Values Distribution (table size: {len(self.Q)})')
        plt.xlabel('Q-Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()