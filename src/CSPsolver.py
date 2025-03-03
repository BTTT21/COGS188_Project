"""
CSPsolver.py

A CSP solver for Minesweeper that uses the grid built in minesweeper.py.
It imports the current game instance (or at least the grid and its dimensions)
from minesweeper.py. Covered cells adjacent to an uncovered cell become CSP variables,
while flagged cells (or those not adjacent to any clue) are fixed. Each uncovered cell
(with a clue) produces a constraint: the sum of the adjacent variable values must equal
the clue (adjusted by any fixed assignments).

Usage (from your Minesweeper project):
    from CSPsolver import MinesweeperCSP, print_csp_solution
    from minesweeper import game  # game should be the current Game instance
    csp_solver = MinesweeperCSP(game.grid, game.squares_y, game.squares_x)
    solution = csp_solver.solve()
    if solution is not None:
        print_csp_solution(game.grid, solution, game.squares_y, game.squares_x)
"""

# Try importing the game instance from minesweeper.py.
try:
    from minesweeper import game
except ImportError:
    game = None  # If not available, a dummy board will be used in __main__

class MinesweeperCSP:
    def __init__(self, grid, rows, cols):
        """
        Initialize the CSP solver using the Minesweeper grid.
        
        :param grid: 2D list of Cell objects from minesweeper.py.
                     Each Cell should have:
                         - is_visible (bool)
                         - bomb_count (int) for uncovered cells
                         - has_bomb (bool)
                         - has_flag (bool)
        :param rows: number of rows in the grid.
        :param cols: number of columns in the grid.
        """
        self.grid = grid
        self.rows = rows
        self.cols = cols

        # Variables: keys are (i,j) coordinates for covered cells that are adjacent to an uncovered cell.
        # Domain for each variable is [0, 1] (0: Safe, 1: Mine).
        self.variables = {}
        # Fixed assignments:
        #   - Covered cells not adjacent to any uncovered cell are fixed as safe (0).
        #   - Flagged cells are fixed as mines (1).
        self.fixed = {}
        # Constraints: list of tuples (target, [list of variable coordinates]).
        # Each uncovered cell with a clue (and not a bomb) produces a constraint.
        self.constraints = []
        # Mapping from each variable to the indices of constraints in which it appears.
        self.var_to_constraints = {}
        
        self.build_csp()

    def build_csp(self):
        # Identify frontier cells: covered cells adjacent to at least one uncovered cell.
        for i in range(self.rows):
            for j in range(self.cols):
                cell = self.grid[i][j]
                if not cell.is_visible:
                    adjacent_to_visible = False
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < self.rows and 0 <= nj < self.cols:
                                if self.grid[ni][nj].is_visible:
                                    adjacent_to_visible = True
                    if adjacent_to_visible:
                        if cell.has_flag:
                            self.fixed[(i, j)] = 1
                        else:
                            self.variables[(i, j)] = [0, 1]
                    else:
                        self.fixed[(i, j)] = 0

        # Build constraints from uncovered cells (with clues).
        for i in range(self.rows):
            for j in range(self.cols):
                cell = self.grid[i][j]
                if cell.is_visible and not cell.has_bomb:
                    target = cell.bomb_count
                    adj_vars = []
                    fixed_sum = 0
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
                                        adj_vars.append((ni, nj))
                    adjusted_target = target - fixed_sum
                    if adj_vars or adjusted_target != 0:
                        self.constraints.append((adjusted_target, adj_vars))
        
        # Build mapping from each variable to the constraints in which it appears.
        self.var_to_constraints = {var: [] for var in self.variables}
        for idx, (target, var_list) in enumerate(self.constraints):
            for var in var_list:
                if var in self.var_to_constraints:
                    self.var_to_constraints[var].append(idx)

    def is_consistent(self, assignment, var):
        """
        For each constraint involving the given variable, check that:
          - The sum of assigned values does not exceed the target.
          - Even if every unassigned variable becomes a mine (1), the target can be reached.
        """
        for idx in self.var_to_constraints.get(var, []):
            target, var_list = self.constraints[idx]
            assigned_sum = 0
            unassigned_count = 0
            for v in var_list:
                if v in assignment:
                    assigned_sum += assignment[v]
                else:
                    unassigned_count += 1
            if assigned_sum > target or assigned_sum + unassigned_count < target:
                return False
        return True

    def select_unassigned_variable(self, assignment):
        """
        Select the next unassigned variable using a simple degree heuristic:
        choose the variable that appears in the most constraints.
        """
        unassigned = [v for v in self.variables if v not in assignment]
        if not unassigned:
            return None
        unassigned.sort(key=lambda v: len(self.var_to_constraints.get(v, [])), reverse=True)
        return unassigned[0]

    def backtracking_search(self, assignment):
        """
        Recursive backtracking search with forward checking.
        """
        if len(assignment) == len(self.variables):
            return assignment
        var = self.select_unassigned_variable(assignment)
        for value in self.variables[var]:
            assignment[var] = value
            if self.is_consistent(assignment, var):
                result = self.backtracking_search(assignment)
                if result is not None:
                    return result
            del assignment[var]
        return None

    def solve(self):
        """
        Solve the Minesweeper CSP.
        
        :return: A dictionary mapping each covered cell coordinate (i, j) to 0 (Safe) or 1 (Mine),
                 including both fixed assignments and those determined via backtracking.
        """
        result = self.backtracking_search({})
        if result is None:
            return None
        full_assignment = {}
        full_assignment.update(self.fixed)
        full_assignment.update(result)
        return full_assignment

def print_csp_solution(grid, assignment, rows, cols):
    """
    Utility function to print the Minesweeper board along with the CSP solution.
    For each cell:
      - Uncovered cells display their bomb_count.
      - Covered cells display:
            "M" if assigned as a mine (1),
            "S" if safe (0),
            "?" if no assignment is available.
    """
    for i in range(rows):
        row_str = []
        for j in range(cols):
            cell = grid[i][j]
            if cell.is_visible:
                row_str.append(str(cell.bomb_count))
            else:
                if (i, j) in assignment:
                    row_str.append("M" if assignment[(i, j)] == 1 else "S")
                else:
                    row_str.append("?")
        print(" ".join(row_str))

if __name__ == "__main__":
    # Use the imported game instance.
    csp_solver = MinesweeperCSP(game.grid, game.squares_y, game.squares_x)
    solution = csp_solver.solve()
    if solution is None:
        print("No solution found.")
    else:
        print("CSP Solution:")
        print_csp_solution(game.grid, solution, game.squares_y, game.squares_x)