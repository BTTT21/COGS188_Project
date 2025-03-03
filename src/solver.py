class BaseSolver:
    def __init__(self, game):
        self.game = game
        
    def select_action(self):
        """选择要执行的动作（返回 (row, col, action_type)）"""
        raise NotImplementedError
        
    def update_model(self, reward, new_state):
        """更新内部模型（如果需要）"""
        pass
        
    def train(self, episodes):
        """训练方法"""
        raise NotImplementedError