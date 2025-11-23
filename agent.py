from typing import List

class TicTacToeAgent:
    def __init__(self, player: int):
        """
        Initialize the TicTacToe agent.
        
        Args:
            player: 1 for X, -1 for O
        """
        self.player = player
    
    def get_action(self, state: List[int], valid_moves: List[int]) -> int:
        raise NotImplementedError("Method not implemented!")
