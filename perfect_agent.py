"""
Perfect TicTacToe Agent - Implements optimal strategy
Adapted from the MrMiyagi agent strategy
"""
import random
from typing import Optional
from agent import TicTacToeAgent


class PerfectAgent(TicTacToeAgent):
    """
    Implements a perfect agent that follows a hardcoded strategy
    to make the best move available at each time.
    """

    def __init__(self, player: int):
        """
        Initialize the perfect agent.
        
        Args:
            player: 1 for X, -1 for O
        """
        super().__init__(player)
        self.level = 0.8

    def _state_to_matrix(self, state: list[int]) -> list[list[int]]:
        """Convert flat state list to 3x3 matrix."""
        return [
            [state[0], state[1], state[2]],
            [state[3], state[4], state[5]],
            [state[6], state[7], state[8]]
        ]

    def _matrix_to_index(self, row: int, col: int) -> int:
        """Convert matrix coordinates to flat index."""
        return row * 3 + col

    def _find_move(self, board: list[list[int]], symbol: int) -> Optional[int]:
        """Helper function to find winning or blocking move."""
        # Check rows
        for i in range(3):
            if board[i].count(symbol) == 2 and board[i].count(0) == 1:
                col = board[i].index(0)
                return self._matrix_to_index(i, col)
        
        # Check columns
        for i in range(3):
            col = [board[j][i] for j in range(3)]
            if col.count(symbol) == 2 and col.count(0) == 1:
                row = col.index(0)
                return self._matrix_to_index(row, i)
        
        # Check diagonal (top-left to bottom-right)
        diag1 = [board[i][i] for i in range(3)]
        if diag1.count(symbol) == 2 and diag1.count(0) == 1:
            index = diag1.index(0)
            return self._matrix_to_index(index, index)
        
        # Check diagonal (top-right to bottom-left)
        diag2 = [board[i][2 - i] for i in range(3)]
        if diag2.count(symbol) == 2 and diag2.count(0) == 1:
            index = diag2.index(0)
            return self._matrix_to_index(index, 2 - index)
        
        return None

    def _win(self, board: list[list[int]]) -> Optional[int]:
        """Find a winning move if one exists."""
        return self._find_move(board, self.player)

    def _block_win(self, board: list[list[int]]) -> Optional[int]:
        """Block opponent's winning move."""
        return self._find_move(board, -self.player)

    def _find_fork(self, board: list[list[int]], symbol: int) -> Optional[int]:
        """Helper function to find fork opportunities."""
        # Try center and corners first
        positions = [(1, 1), (0, 0), (0, 2), (2, 0), (2, 2)]
        
        for i, j in positions:
            if board[i][j] == 0:
                # Temporarily place symbol
                board[i][j] = symbol
                
                # Count how many ways we can win from this position
                winning_moves = 0
                for _ in range(4):  # Check multiple times for different winning lines
                    if self._find_move(board, symbol) is not None:
                        winning_moves += 1
                
                # Restore board
                board[i][j] = 0
                
                # If we can win in 2+ ways, it's a fork
                if winning_moves >= 2:
                    return self._matrix_to_index(i, j)
        
        return None

    def _fork(self, board: list[list[int]]) -> Optional[int]:
        """Create a fork opportunity."""
        return self._find_fork(board, self.player)

    def _block_fork(self, board: list[list[int]]) -> Optional[int]:
        """Block opponent's fork opportunity."""
        return self._find_fork(board, -self.player)

    def _center(self, board: list[list[int]]) -> Optional[int]:
        """Take the center if available."""
        if board[1][1] == 0:
            return 4
        return None

    def _corner(self, board: list[list[int]]) -> Optional[int]:
        """Take an available corner."""
        for i, j in [(0, 0), (0, 2), (2, 0), (2, 2)]:
            if board[i][j] == 0:
                return self._matrix_to_index(i, j)
        return None

    def _side_empty(self, board: list[list[int]]) -> Optional[int]:
        """Take an available side."""
        for i, j in [(0, 1), (1, 0), (1, 2), (2, 1)]:
            if board[i][j] == 0:
                return self._matrix_to_index(i, j)
        return None

    def get_action(self, state: list[int], valid_moves: list[int], training=True) -> int:
        """
        Select the best move using optimal strategy.
        
        Args:
            state: Current game state as flat list
            valid_moves: List of valid move indices
            
        Returns:
            Index of chosen move
        """
        board = self._state_to_matrix(state)
        
        # With probability (1 - level), make a random move
        if random.random() > self.level:
            return random.choice(valid_moves)
        
        # Apply strategy in order of priority
        strategies = [
            self._win,           # Win if possible
            self._block_win,     # Block opponent's win
            self._fork,          # Create fork
            self._block_fork,    # Block opponent's fork
            self._center,        # Take center
            self._corner,        # Take corner
            self._side_empty,    # Take side
        ]
        
        for strategy in strategies:
            action = strategy(board)
            if action is not None and action in valid_moves:
                return action
        
        # Fallback to random move
        return random.choice(valid_moves)

    def __str__(self):
        return f"PerfectAgent(level={self.level})"
