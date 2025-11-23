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
    new_state[position] = player
    return new_state

def play_game(agent: TicTacToeAgent, opponent: TicTacToeAgent, training=True):
    """Play one game and return reward"""
    state = [0] * 9
    current_player = X
    episode_experiences = []  # Store all experiences for this episode
    
    while True:
        valid_moves = get_valid_moves(state)
        
        if current_player == agent.player:
            action = agent.get_action(state, valid_moves, training)
            new_state = make_move(state, action, current_player)
            
            result = has_won(new_state, agent.player)
            
            # Assign rewards based on game outcome
            if result is True:
                reward = 1.0
                done = True
            elif result is False:
                reward = -1.0
                done = True
            elif result == "draw":
                reward = 0.5
                done = True
            else:
                # Small negative reward for continuing (encourages faster wins)
                reward = -0.01
                done = False
            
            if training:
                # Store this experience
                episode_experiences.append((state.copy(), action, reward, new_state.copy(), done))
            
            state = new_state
            
            if done:
                # Backpropagate rewards through all agent's moves
                if training:
                    for i, (s, a, r, ns, d) in enumerate(episode_experiences):
                        # Give final reward to all moves (temporal credit assignment)
                        if i == len(episode_experiences) - 1:
                            agent.remember(s, a, reward, ns, d)
                        else:
                            # Intermediate moves get discounted reward
                            agent.remember(s, a, -0.01, ns, False)
                    
                    # Train after EACH game
                    agent.replay()
                
                return reward
        else:
            # Opponent's turn
            action = opponent.get_action(state, valid_moves)
            
            prev_state = state.copy()
            state = make_move(state, action, current_player)
            
            result = has_won(state, agent.player)
            
            # Store experiences when opponent's move ends the game
            if training and result is not None and len(episode_experiences) > 0:
                # Opponent won or caused draw - update last agent move with this info
                if result is False:
                    # Agent lost due to opponent's move
                    last_state, last_action, _, last_next, _ = episode_experiences[-1]
                    episode_experiences[-1] = (last_state, last_action, -1.0, state.copy(), True)
                elif result == "draw":
                    last_state, last_action, _, last_next, _ = episode_experiences[-1]
                    episode_experiences[-1] = (last_state, last_action, 0.5, state.copy(), True)
            
            if result is not None:
                if training:
                    # Store all experiences with final outcome
                    final_reward = 1.0 if result is True else (-1.0 if result is False else 0.5)
                    for i, (s, a, r, ns, d) in enumerate(episode_experiences):
                        if i == len(episode_experiences) - 1:
                            agent.remember(s, a, final_reward, state.copy(), True)
                        else:
                            agent.remember(s, a, -0.01, ns, False)
                    
                    agent.replay()
                
                if result is True:
                    return 1.0
                elif result is False:
                    return -1.0
                else:
                    return 0.5
        
        current_player = -current_player


def train_agent(opponent: TicTacToeAgent, episodes=10000, update_target_every=100):
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
        reward = play_game(agent, opponent, training=True)
        rewards_history.append(reward)
        
        # Track outcomes
        if reward > 0.9:
            wins += 1
        elif reward < -0.9:
            losses += 1
        else:
            draws += 1
        
        # Update target network periodically
        if episode % update_target_every == 0:
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

def invalid_move(pre: list[int], after: list[int]) -> bool:
    """Check if move is invalid"""
    return (
        len(list(filter(lambda x: x == 0, pre)))
        - len(list(filter(lambda x: x == 0, after)))
        != 1
    )


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
