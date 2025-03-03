import numpy as np
from itertools import product

class Minesweeper:
    def __init__(self, rows=16, cols=32, mines=99):
        self.rows = rows
        self.cols = cols
        self.total_mines = mines
        self.reset()
        
    def reset(self, first_click=None):
        """初始化游戏板，确保第一次点击不是地雷"""
        self.board = np.zeros((self.rows, self.cols), dtype=int)  # -1: 雷, 0-8: 数字
        self.revealed = np.full((self.rows, self.cols), False)
        self.flags = np.full((self.rows, self.cols), False)
        self.game_over = False
        self.victory = False
        self._place_mines(first_click)
        self._calculate_numbers()
        
    def _place_mines(self, safe_cell):
        """随机放置地雷，排除安全区域"""
        candidates = list(product(range(self.rows), range(self.cols)))
        if safe_cell:
            candidates.remove(safe_cell)
            neighbors = self._get_neighbors(*safe_cell)
            candidates = [c for c in candidates if c not in neighbors]
            
        mine_positions = np.random.choice(
            len(candidates), self.total_mines, replace=False)
        for idx in mine_positions:
            r, c = candidates[idx]
            self.board[r, c] = -1

    def _calculate_numbers(self):
        """计算每个单元格周围的地雷数量"""
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r, c] != -1:
                    self.board[r, c] = self._count_mines(r, c)
    
    def _count_mines(self, row, col):
        """计算指定单元格周围的地雷数量"""
        return sum(
            1 for nr, nc in self._get_neighbors(row, col)
            if self.board[nr, nc] == -1
        )

    def _get_neighbors(self, row, col):
        """获取相邻单元格坐标"""
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    neighbors.append((nr, nc))
        return neighbors

    def reveal(self, row, col):
        """揭示单元格并处理级联效果"""
        if self.game_over or self.flags[row, col]:
            return False
            
        if not self.revealed[row, col]:
            if self.board[row, col] == -1:  # 踩雷
                self.game_over = True
                self.victory = False
                return False
                
            self.revealed[row, col] = True
            if self.board[row, col] == 0:  # 级联揭示
                self._cascade_reveal(row, col)
                
        return self.check_victory()

    def _cascade_reveal(self, row, col):
        """级联揭示零值区域"""
        stack = [(row, col)]
        while stack:
            r, c = stack.pop()
            for nr, nc in self._get_neighbors(r, c):
                if not self.revealed[nr, nc] and not self.flags[nr, nc]:
                    self.revealed[nr, nc] = True
                    if self.board[nr, nc] == 0:
                        stack.append((nr, nc))

    def toggle_flag(self, row, col):
        """切换标记状态"""
        if not self.revealed[row, col] and not self.game_over:
            self.flags[row, col] = not self.flags[row, col]
        return self.check_victory()

    def check_victory(self):
        """检查是否胜利"""
        if np.sum(self.revealed) == self.rows*self.cols - self.total_mines:
            self.game_over = True
            self.victory = True
        return self.victory

    def get_observation(self):
        """返回当前游戏状态（供AI使用）"""
        return {
            'board': self.board.copy(),
            'revealed': self.revealed.copy(),
            'flags': self.flags.copy(),
            'game_over': self.game_over,
            'victory': self.victory
        }

    def render(self):
        """打印当前游戏状态（调试用）"""
        symbols = {
            -2: 'F',  # 标记
            -1: 'X',  # 未揭示雷
            0: ' ',    # 空白
        }
        for r in range(self.rows):
            row_str = []
            for c in range(self.cols):
                if self.flags[r, c]:
                    row_str.append(symbols[-2])
                elif not self.revealed[r, c]:
                    row_str.append('.')
                else:
                    val = self.board[r, c]
                    row_str.append(str(val) if val > 0 else symbols.get(val, ' '))
            print(' '.join(row_str))
        print('Status:', 'Won' if self.victory else 'Lost' if self.game_over else 'Playing')

# 示例用法
if __name__ == "__main__":
    game = Minesweeper()
    game.reveal(8, 16)  # 中间位置首次点击
    game.render()