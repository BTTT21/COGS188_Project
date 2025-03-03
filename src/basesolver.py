class BaseSolver:
    def __init__(self, game):
        self.game = game
        
    def select_action(self):
        raise NotImplementedError
        
    def update_model(self, reward, new_state):
        pass
        
    def train(self, episodes):
        raise NotImplementedError