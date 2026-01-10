from agent import TicTacToeAgent
from constants import X, O
from train import get_valid_moves, has_won, make_move
from utils import print_game_state, print_game_state_with_numbers

from random import choice


def play_against_human(agent: TicTacToeAgent):
    """Play against the trained agent"""
    print("\n=== Play Against AI ===")
    print("Use numbers 1-9 to select positions:")
    print_game_state_with_numbers([0] * 9)
    
    state = [0] * 9
    agent.player = choice([X, O])
    current_player = choice([X, O])
    
    while True:
        print_game_state(state)
        
        if current_player == agent.player:
            print("AI's turn...")
            valid_moves = get_valid_moves(state)
            action = agent.get_action(state, valid_moves, training=False)
            state = make_move(state, action, current_player)[0]
            print(f"AI chose position {action + 1}")
        else:
            print(f"Your turn ({'O' if agent.player == X else 'X'}):")
            print_game_state_with_numbers(state)
            valid_moves = get_valid_moves(state)
            
            while True:
                try:
                    move = int(input(f"Enter position (1-9): ")) - 1
                    if move in valid_moves:
                        break
                    else:
                        print("Invalid move! Position already taken or out of range.")
                except ValueError:
                    print("Please enter a number between 1 and 9.")
            
            state = make_move(state, move, current_player)[0]
        
        result = has_won(state, agent.player)
        if result is True:
            print_game_state(state)
            print(f"AI ({'O' if agent.player == O else 'X'}) wins!")
            break
        elif result is False:
            print_game_state(state)
            print(f"You ({'O' if agent.player == X else 'X'}) win!")
            break
        elif result == "draw":
            print_game_state(state)
            print("It's a draw!")
            break
        
        current_player = -current_player
