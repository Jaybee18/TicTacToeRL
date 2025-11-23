"""
Random TicTacToe Agent - Makes random moves
"""
import random
from agent import TicTacToeAgent


class RandomAgent(TicTacToeAgent):
    """
    Implements a simple agent that makes random valid moves.
    """

    def __init__(self, player: int):
        """
        Initialize the random agent.
        
        Args:
            player: 1 for X, -1 for O
        """
        self.player = player

    def get_action(self, state: list[int], valid_moves: list[int]) -> int:
        """
        Select a random valid move.
        
        Args:
            state: Current game state as flat list (not used)
            valid_moves: List of valid move indices
            
        Returns:
            Index of randomly chosen move
        """
        return random.choice(valid_moves)

    def __str__(self):
        return "RandomAgent"
