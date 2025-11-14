from enum import Enum
from random import randint

import keras
import numpy as np

game_state: list[int] = [0, 0, 0, 0, 0, 0, 0, 0, 0]

X = 1
O = -1


def print_game_state(state: list):
    print(
        """
[{}, {}, {}]
[{}, {}, {}]
[{}, {}, {}]
""".format(*["X" if s == X else "O" if s == O else " " for s in state])
    )


def create_q_model():
    return keras.Sequential(
        [
            keras.layers.Dense(32, activation="relu", input_shape=(9,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(9, activation="linear"),
            keras.layers.Softmax(),
        ]
    )


def has_won(state: list, player: int) -> bool | None:
    """Checks if player has won or lost returning True or False. Returns None if game is ongoing"""
    if state[0] == state[1] == state[2] != 0:
        return state[0] == player
    elif state[3] == state[4] == state[5] != 0:
        return state[3] == player
    elif state[6] == state[7] == state[8] != 0:
        return state[6] == player
    elif state[0] == state[3] == state[6] != 0:
        return state[0] == player
    elif state[1] == state[4] == state[7] != 0:
        return state[1] == player
    elif state[2] == state[5] == state[8] != 0:
        return state[2] == player
    elif state[0] == state[4] == state[8] != 0:
        return state[0] == player
    elif state[2] == state[4] == state[6] != 0:
        return state[2] == player
    elif 0 not in state:
        return True
    return None


def game_loop():
    global game_state

    turn = -1

    while True:
        game_state[randint(0, 8)] = turn
        print_game_state(game_state)

        if has_won(game_state, turn):
            print("Player", "X" if turn == X else "O", "wins!")
            break

        turn = -turn

    print("Game Over!")


def invalid_move(pre: list[int], after: list[int]) -> bool:
    return (
        len(list(filter(lambda x: x == 0, after)))
        - len(list(filter(lambda x: x == 0, pre)))
        == 1
    )


def main():
    print("Hello from tictactoerl!")

    # print_game_state(game_state)

    # model = create_q_model()
    # print(model.predict(np.array([game_state])))
    #
    game_loop()


if __name__ == "__main__":
    main()
