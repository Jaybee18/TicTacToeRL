from matplotlib import pyplot as plt
from random import choice
import numpy as np

from agent import TicTacToeAgent
from constants import O, X
from dqn_agent import DQNAgent


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

def make_move(state: list[int], position: int, player: int) -> tuple[list[int], float, bool]:
    """Returns new state with move applied"""
    new_state = state.copy()

    # No changes to the state on illegal moves
    if new_state[position] != 0:
        pass
    else:
        new_state[position] = player

    # Calculate reward
    result = has_won(new_state, player)
    reward = get_reward(state, new_state, result)

    # Determine if the game is over
    done = result is True or result is False or result == "draw" or state.count(0) == new_state.count(0)

    return new_state, reward, done

def get_reward(old_state: list[int], new_state: list[int], result) -> float:
    # Illegal moves are seen as a loss
    if old_state.count(0) == new_state.count(0):
        return -1
    
    # Winning and losing is 1 and -1 respectively
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
        new_state, reward, done = make_move(state, action, cp.player)

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
            won = None if reward == 0 else cp.player if reward == 1 else -cp.player
            break

    return episode_experiences_x, episode_experiences_o, won

def train_agent(agent: TicTacToeAgent, opponent: TicTacToeAgent, episodes=10000, update_target_every=250):
    """Train the DQN agent"""
    agent = DQNAgent()
    agent.episodes_trained = episodes  # Store total episodes for filename
    rewards_history = []
    avg_rewards = []
    wins = 0
    losses = 0
    draws = 0
    
    print("Training started...")

    for episode in range(episodes):
        agent.player = choice([X, O])
        opponent.player = -agent.player

        exp_x, exp_o, winner = play_game(agent, opponent, training=True)
        exp = exp_x if agent.player == X else exp_o
        
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
        for e in exp:
            agent.remember(*e)

        # Train the agents network for one step
        agent.replay()

        # Update the target network less frequently to have some
        # stability during training
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
