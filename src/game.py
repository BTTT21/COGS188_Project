import numpy as np
from itertools import product

class Minesweeper:
    def __init__(self, rows=16, cols=32, mines=99):
        self.rows = rows
        self.cols = cols
        self.total_mines = mines
        self.first_click = True  # 标识是否是第一次点击
        # 初始化棋盘，暂时不放置地雷
        self.board = np.zeros((self.rows, self.cols), dtype=int)  # -1: 雷, 0-8: 数字
        self.revealed = np.full((self.rows, self.cols), False)
        self.flags = np.full((self.rows, self.cols), False)
        self.game_over = False
        self.victory = False

    def reset(self):
        """重置游戏状态，不直接放置地雷，等待首次点击"""
        self.first_click = True
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.revealed = np.full((self.rows, self.cols), False)
        self.flags = np.full((self.rows, self.cols), False)
        self.game_over = False
        self.victory = False

    def _place_mines(self, safe_cell):
        """随机放置地雷，排除安全区域（safe_cell 及其邻居）"""
        candidates = list(product(range(self.rows), range(self.cols)))
        if safe_cell:
            if safe_cell in candidates:
                candidates.remove(safe_cell)
            neighbors = self._get_neighbors(*safe_cell)
            candidates = [c for c in candidates if c not in neighbors]
        if self.total_mines > len(candidates):
            raise ValueError("候选位置不足以放置所有地雷。")
        mine_positions = np.random.choice(len(candidates), self.total_mines, replace=False)
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
        # 检查输入是否在范围内
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            print("输入坐标超出范围。")
            return False
        if self.game_over or self.flags[row, col]:
            return False

        # 第一次点击时生成雷区，确保安全区域无雷
        if self.first_click:
            self._place_mines((row, col))
            self._calculate_numbers()
            self.first_click = False

        if not self.revealed[row, col]:
            if self.board[row, col] == -1:  # 踩雷
                self.revealed[row, col] = True  # 揭示触雷的单元格
                self.game_over = True
                self.victory = False
                self._reveal_all_mines()  # 揭示所有地雷
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

    def _reveal_all_mines(self):
        """游戏结束时揭示所有地雷"""
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r, c] == -1:
                    self.revealed[r, c] = True

    def toggle_flag(self, row, col):
        """切换标记状态"""
        if not self.revealed[row, col] and not self.game_over:
            self.flags[row, col] = not self.flags[row, col]
        return self.check_victory()

    def check_victory(self):
        """检查是否胜利"""
        if np.sum(self.revealed) == self.rows * self.cols - self.total_mines:
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
            -1: 'X',  # 地雷
            0: ' ',   # 空白
            -2: 'F',  # 标记
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
    # 首次点击（例如中间位置）
    game.reveal(8, 16)
    
    # 进入交互式循环，直到游戏结束
    while not game.game_over:
        game.render()
        cmd = input("请输入操作 (r 行 列 来揭示，f 行 列 来标记): ")
        parts = cmd.split()
        if len(parts) != 3:
            print("输入格式错误，请按提示输入，例如：r 8 16")
            continue
        
        action, row_str, col_str = parts
        try:
            row = int(row_str)
            col = int(col_str)
        except ValueError:
            print("行和列需要是整数。")
            continue

        # 检查输入坐标是否在范围内
        if not (0 <= row < game.rows and 0 <= col < game.cols):
            print("输入坐标超出范围，请输入0到{}之间的数字。".format(game.rows - 1))
            continue
        
        if action.lower() == 'r':
            game.reveal(row, col)
        elif action.lower() == 'f':
            game.toggle_flag(row, col)
        else:
            print("未知操作。请使用 r（揭示）或 f（标记）。")
    
    # 游戏结束后，展示最终状态
    game.render()
    if game.victory:
        print("恭喜你，获胜了！")
    else:
        print("游戏结束，你输了。")