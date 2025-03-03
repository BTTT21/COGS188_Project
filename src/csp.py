import numpy as np
import time
from collections import defaultdict

class CSPSolver(BaseSolver):
    def __init__(self, game):
        super().__init__(game)
        self.constraints = []
        
    def build_constraints(self):
        self.constraints = []
        for r in range(self.game.rows):
            for c in range(self.game.cols):
                if self.game.revealed[r, c] and self.game.board[r, c] > 0:
                    neighbors = self._get_neighbors(r, c)
                    hidden = [(nr, nc) for (nr, nc) in neighbors 
                             if not self.game.revealed[nr, nc]]
                    self.constraints.append({
                        'cells': hidden,
                        'required': self.game.board[r, c] - np.sum(self.game.flags[nr, nc] for (nr, nc) in neighbors)
                    })
        
    def backtrack_search(self):
        variables = list({cell for constr in self.constraints for cell in constr['cells']})
        assignment = {}
        return self._backtrack(assignment, variables)
        
    def _backtrack(self, assignment, variables):
        if not variables:
            return assignment
        var = variables[0]
        for value in [0, 1]:  # 0: Safe, 1: Mine
            if self._is_consistent(var, value, assignment):
                assignment[var] = value
                result = self._backtrack(assignment.copy(), variables[1:])
                if result is not None:
                    return result
                del assignment[var]
        return None
        
    def _is_consistent(self, var, value, assignment):
        temp_assignment = assignment.copy()
        temp_assignment[var] = value
        for constr in self.constraints:
            relevant = [temp_assignment.get(cell, 0) for cell in constr['cells']]
            total = sum(relevant)
            if total > constr['required'] or (len(relevant) - sum(x is None for x in relevant)) < constr['required'] - total:
                return False
        return True
        
    def select_action(self):
        self.build_constraints()
        solution = self.backtrack_search()
        if solution:
            safe = [cell for cell, val in solution.items() if val == 0]
            if safe:
                r, c = safe[0]
                return (r, c, 'reveal')
        # Fallback to random
        candidates = np.argwhere(~self.game.revealed & ~self.game.flags)
        if candidates:
            r, c = candidates[0]
            return (r, c, 'reveal')
        return None
        
    def train(self, episodes):
        for _ in range(episodes):
            self.game.reset()
            while not self.game.game_over:
                action = self.select_action()
                if not action:
                    break
                r, c, _ = action
                self.game.reveal(r, c)
