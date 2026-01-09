from constants import O, X


def print_game_state(state: list):
    print(
        """
[{}, {}, {}]
[{}, {}, {}]
[{}, {}, {}]
""".format(*["X" if s == X else "O" if s == O else " " for s in state])
    )

def print_game_state_with_numbers(state: list):
    """Print board with numbers for empty spaces"""
    display = []
    for i, s in enumerate(state):
        if s == X:
            display.append("X")
        elif s == O:
            display.append("O")
        else:
            display.append(str(i + 1))
    
    print(
        """
[{}, {}, {}]
[{}, {}, {}]
[{}, {}, {}]
""".format(*display)
    )
