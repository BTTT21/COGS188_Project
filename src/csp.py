import numpy as np
import time
from collections import defaultdict
from solver import BaseSolver

class CSPSolver(BaseSolver):
    def __init__(self, game):
        super().__init__(game)
        self.constraints = []
        
    def build_constraints(self):
        self.constraints = []
        for r in range(self.game.rows):
            for c in range(self.game.cols):
                # 只对已揭示且数字大于0的格子构建约束
                if self.game.revealed[r, c] and self.game.board[r, c] > 0:
                    neighbors = self.game._get_neighbors(r, c)
                    hidden = [(nr, nc) for (nr, nc) in neighbors if not self.game.revealed[nr, nc]]
                    if not hidden:
                        continue
                    # 计算该格子周围已标记的数量
                    flagged_count = sum(1 for (nr, nc) in neighbors if self.game.flags[nr, nc])
                    required = self.game.board[r, c] - flagged_count
                    # 确保约束合理
                    if 0 <= required <= len(hidden):
                        self.constraints.append({
                            'cells': hidden,
                            'required': required
                        })
        
    def backtrack_search(self):
        # 收集所有出现在约束中的变量
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
            # 计算该约束中已赋值的和未赋值的变量个数
            assigned = [temp_assignment[cell] for cell in constr['cells'] if cell in temp_assignment]
            sum_assigned = sum(assigned)
            unassigned = [cell for cell in constr['cells'] if cell not in temp_assignment]
            required = constr['required']
            # 已分配的地雷数量不应超过要求
            if sum_assigned > required:
                return False
            # 即使所有未赋值的都为地雷，也必须能够达到要求
            if sum_assigned + len(unassigned) < required:
                return False
        return True
        
    def select_action(self):
        self.build_constraints()
        solution = self.backtrack_search()
        if solution:
            # 优先选择被推断为安全的格子
            safe_cells = [cell for cell, val in solution.items() if val == 0]
            if safe_cells:
                r, c = safe_cells[0]
                return (r, c, 'reveal')
        # 若没有推断出安全动作，则退回到随机选择
        candidates = np.argwhere(~self.game.revealed & ~self.game.flags)
        if candidates.size > 0:
            r, c = candidates[0]
            return (r, c, 'reveal')
        return None
        
    def train(self, episodes):
        for _ in range(episodes):
            self.game.reset()
            while not self.game.game_over:
                action = self.select_action()
                if action is None:
                    break
                r, c, _ = action
                self.game.reveal(r, c)