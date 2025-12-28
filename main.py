import os
import glob
from enum import Enum
from random import randint, random, sample
import matplotlib.pyplot as plt
import numpy as np

from perfect_agent import PerfectAgent
from dqn_agent import DQNAgent
from agent import TicTacToeAgent

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

def has_won(state: list, player: int) -> bool | None:
    """Checks if player has won returning True, lost returning False, draw returning 'draw'. Returns None if game is ongoing"""
    # Check rows
    if state[0] == state[1] == state[2] != 0:
        return state[0] == player
    elif state[3] == state[4] == state[5] != 0:
        return state[3] == player
    elif state[6] == state[7] == state[8] != 0:
        return state[6] == player
    # Check columns
    elif state[0] == state[3] == state[6] != 0:
        return state[0] == player
    elif state[1] == state[4] == state[7] != 0:
        return state[1] == player
    elif state[2] == state[5] == state[8] != 0:
        return state[2] == player
    # Check diagonals
    elif state[0] == state[4] == state[8] != 0:
        return state[0] == player
    elif state[2] == state[4] == state[6] != 0:
        return state[2] == player
    # Check draw
    elif 0 not in state:
        return "draw"
    return None


def get_valid_moves(state: list[int]) -> list[int]:
    """Returns list of valid move indices"""
    return [i for i, val in enumerate(state) if val == 0]


def make_move(state: list[int], position: int, player: int) -> list[int]:
    """Returns new state with move applied"""
    new_state = state.copy()

    # No changes to the state on illegal moves
    if new_state[position] != 0:
        return new_state

    new_state[position] = player
    return new_state

def get_reward(old_state: list[int], new_state: list[int], result) -> float:
    # Illegal moves are seen as a loss
    if old_state.count(0) == new_state.count(0):
        return -1
    
    # Winning and losing is 5 and -5 respectively
    if result is True:
        return 1
    if result is False:
        return -1
    
    # Draws are the best outcome when playing against the perfect player
    if result == "draw":
        return 0
    
    # No reward for continuing
    return 0

def play_game(agent: TicTacToeAgent, opponent: TicTacToeAgent, training=True):
    """Play one game and return reward"""
    state = [0] * 9
    current_player = X
    episode_experiences_x = []  # Store all experiences for this episode
    episode_experiences_o = []  # Store all experiences for this episode
    won = None
    
    while True:
        cp = agent if agent.player == current_player else opponent
        
        valid_moves = get_valid_moves(state) # Not really needed
        action = cp.get_action(state, valid_moves, training)

        # Make env step
        new_state = make_move(state, action, cp.player)
        result = has_won(new_state, cp.player)
        reward = get_reward(state, new_state, result)
        done = result is True or result is False or result == "draw" or state.count(0) == new_state.count(0)

        # Store step as memory
        if cp.player == X:
            episode_experiences_x.append([state.copy(), action, reward, new_state.copy(), done])
        else:
            episode_experiences_o.append([state.copy(), action, reward, new_state.copy(), done])
        
        # Retroactively give a negative reward for the losing agent 
        # if the current move ended the game
        if done and reward == 1:
            # The winning player is cp, so update the OTHER player's last move
            if cp.player == X and len(episode_experiences_o) > 0:
                episode_experiences_o[-1][2] = -1
            elif cp.player == O and len(episode_experiences_x) > 0:
                episode_experiences_x[-1][2] = -1
        
        # Commit new state and change player
        state = new_state
        current_player = -current_player

        if done:
            won = None if result == "draw" else cp.player if result else -cp.player
            break

    return episode_experiences_x, episode_experiences_o, won


def train_agent(opponent: TicTacToeAgent, episodes=10000, update_target_every=250):
    """Train the DQN agent"""
    agent = DQNAgent(player=X)
    agent.episodes_trained = episodes  # Store total episodes for filename
    rewards_history = []
    avg_rewards = []
    wins = 0
    losses = 0
    draws = 0
    
    print("Training started...")

    for episode in range(episodes):
        exp_x, exp_o, winner = play_game(agent, opponent, training=True)
        
        # Track outcomes
        if winner == agent.player:
            wins += 1
            rewards_history.append(1)
        elif winner == None:
            draws += 1
            rewards_history.append(0)
        else:
            losses += 1
            rewards_history.append(-1)

        # Store experiences in replay memory for training agent
        for exp in exp_x:
            agent.remember(*exp)

        # Train the agents network for one step
        agent.replay()

        if episode % update_target_every == 0 and episode > 0:
            agent.update_target_model()
        
        # Print progress
        if episode % 100 == 0 and episode > 0:
            avg_reward = np.mean(rewards_history[-100:])
            avg_rewards.append(avg_reward)
            win_rate = wins / 100.0
            loss_rate = losses / 100.0
            draw_rate = draws / 100.0
            print(f"Episode {episode}/{episodes}, Avg Reward: {avg_reward:.3f}, Epsilon: {agent.epsilon:.3f}")
            print(f"  Win: {win_rate:.1%}, Loss: {loss_rate:.1%}, Draw: {draw_rate:.1%}")
            wins = losses = draws = 0
    
    print("Training completed!")
    
    # Save the trained model
    agent.save()
    
    # Plot rewards
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards_history, alpha=0.3, label='Episode Reward')
    plt.plot([i * 100 for i in range(len(avg_rewards))], avg_rewards, 'r-', linewidth=2, label='Avg Reward (100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    window = 500
    moving_avg = [np.mean(rewards_history[max(0, i-window):i+1]) for i in range(len(rewards_history))]
    plt.plot(moving_avg)
    plt.xlabel('Episode')
    plt.ylabel('Moving Average Reward')
    plt.title(f'Moving Average (window={window})')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('c:\\GitHub\\TicTacToeRL\\training_results.png')
    plt.show()
    
    return agent


def play_against_human(agent: TicTacToeAgent):
    """Play against the trained agent"""
    print("\n=== Play Against AI ===")
    print("You are O, AI is X")
    print("Use numbers 1-9 to select positions:")
    print_game_state_with_numbers([0] * 9)
    
    state = [0] * 9
    current_player = X
    
    while True:
        print_game_state(state)
        
        if current_player == agent.player:
            print("AI's turn...")
            valid_moves = get_valid_moves(state)
            action = agent.get_action(state, valid_moves, training=False)
            state = make_move(state, action, current_player)
            print(f"AI chose position {action + 1}")
        else:
            print("Your turn (O):")
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
            
            state = make_move(state, move, current_player)
        
        result = has_won(state, agent.player)
        if result is True:
            print_game_state(state)
            print("AI (X) wins!")
            break
        elif result is False:
            print_game_state(state)
            print("You (O) win!")
            break
        elif result == "draw":
            print_game_state(state)
            print("It's a draw!")
            break
        
        current_player = -current_player


def main():
    print("=== TicTacToe Deep Q-Learning ===")
    print("1. Train new agent")
    print("2. Play against AI (quick training)")
    print("3. Load trained agent and play")
    
    choice = input("Choose option (1, 2, or 3): ")
    
    opponent = PerfectAgent(player=O)
    if choice == "1":
        episodes = int(input("Enter number of training episodes (default 10000): ") or "10000")
        agent = train_agent(opponent=opponent, episodes=episodes)
        
        play_more = input("\nDo you want to play against the trained agent? (y/n): ")
        if play_more.lower() == 'y':
            while True:
                play_against_human(agent)
                again = input("\nPlay again? (y/n): ")
                if again.lower() != 'y':
                    break
    elif choice == "2":
        print("\nQuick training for demonstration (2500 episodes)...")
        agent = train_agent(opponent=opponent, episodes=2500)
        
        while True:
            play_against_human(agent)
            again = input("\nPlay again? (y/n): ")
            if again.lower() != 'y':
                break
    elif choice == "3":
        models_dir = "c:\\GitHub\\TicTacToeRL\\models"
        model_files = glob.glob(os.path.join(models_dir, "model_*.keras"))
        
        if not model_files:
            print(f"No trained models found in {models_dir}")
            print("Please train a model first (option 1 or 2)")
            return
        
        print("\nAvailable models:")
        for i, model_file in enumerate(model_files, 1):
            print(f"{i}. {os.path.basename(model_file)}")
        
        model_choice = int(input(f"\nSelect model (1-{len(model_files)}): ")) - 1
        model_path = model_files[model_choice]
        
        agent = DQNAgent(player=X)
        agent.load(model_path)
        agent.epsilon = 0.0  # No exploration when playing
        
        print("\nLoaded trained agent. Ready to play!")
        while True:
            play_against_human(agent)
            again = input("\nPlay again? (y/n): ")
            if again.lower() != 'y':
                break
    else:
        print("Invalid choice. Please run again and select 1, 2, or 3.")


if __name__ == "__main__":
    main()
