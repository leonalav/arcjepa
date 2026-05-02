from .portfolio import PortfolioSolver


class ModelGuidedSolver(PortfolioSolver):
    def __init__(self, checkpoint=None, max_steps: int = 100, seed: int = 0):
        super().__init__(max_steps=max_steps, seed=seed)
        self.checkpoint = checkpoint
