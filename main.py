from enum import Enum
from random import randint, random, sample
from collections import deque
import keras
import numpy as np
import matplotlib.pyplot as plt

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


def create_q_model():
    return keras.Sequential(
        [
            keras.layers.Dense(64, activation="relu", input_shape=(9,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(9, activation="linear"),
        ]
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


class DQNAgent:
    def __init__(self, player: int):
        self.player = player
        self.opponent = -player
        self.model = create_q_model()
        self.target_model = create_q_model()
        self.target_model.set_weights(self.model.get_weights())
        
        self.optimizer = keras.optimizers.Adam(learning_rate=0.001)
        self.loss_fn = keras.losses.MeanSquaredError()

        self.model.compile(optimizer=self.optimizer, loss=self.loss_fn)
        self.target_model.compile(optimizer=self.optimizer, loss=self.loss_fn)
        
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32
    
    def get_action(self, state: list[int], valid_moves: list[int], training=True) -> int:
        """Select action using epsilon-greedy policy"""
        if training and random() < self.epsilon:
            return np.random.choice(valid_moves)
        
        q_values = self.model.predict(np.array([state]), verbose=0)[0]
        
        # Mask invalid moves
        masked_q = np.full(9, -np.inf)
        for move in valid_moves:
            masked_q[move] = q_values[move]
        
        return np.argmax(masked_q)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Train on batch from memory"""
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = sample(self.memory, self.batch_size)
        
        states = np.array([exp[0] for exp in minibatch])
        actions = np.array([exp[1] for exp in minibatch])
        rewards = np.array([exp[2] for exp in minibatch])
        next_states = np.array([exp[3] for exp in minibatch])
        dones = np.array([exp[4] for exp in minibatch])
        
        # Get current Q values
        current_q = self.model.predict(states, verbose=0)
        
        # Get next Q values from target model
        next_q = self.target_model.predict(next_states, verbose=0)
        
        # Update Q values
        for i in range(self.batch_size):
            if dones[i]:
                current_q[i][actions[i]] = rewards[i]
            else:
                current_q[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q[i])
        
        # Train model
        self.model.fit(states, current_q, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_model(self):
        """Copy weights from model to target model"""
        self.target_model.set_weights(self.model.get_weights())
    
    def save(self, filepath: str = "c:\\GitHub\\TicTacToeRL\\models\\tictactoe_dqn.keras"):
        """Save the model weights to a file."""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str = "c:\\GitHub\\TicTacToeRL\\models\\tictactoe_dqn.keras"):
        """Load the model weights from a file."""
        self.model = keras.models.load_model(filepath)
        self.target_model.set_weights(self.model.get_weights())
        print(f"Model loaded from {filepath}")


def play_game(agent: DQNAgent, opponent_random=True, training=True):
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
            if opponent_random:
                action = np.random.choice(valid_moves)
            else:
                action = np.random.choice(valid_moves)
            
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


def train_agent(episodes=10000, update_target_every=100):
    """Train the DQN agent"""
    agent = DQNAgent(player=X)
    rewards_history = []
    avg_rewards = []
    wins = 0
    losses = 0
    draws = 0
    
    print("Training started...")
    
    for episode in range(episodes):
        reward = play_game(agent, opponent_random=True, training=True)
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


def play_against_human(agent: DQNAgent):
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


def game_loop():
    # ...existing code...
    pass


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
    
    if choice == "1":
        episodes = int(input("Enter number of training episodes (default 10000): ") or "10000")
        agent = train_agent(episodes=episodes)
        
        play_more = input("\nDo you want to play against the trained agent? (y/n): ")
        if play_more.lower() == 'y':
            while True:
                play_against_human(agent)
                again = input("\nPlay again? (y/n): ")
                if again.lower() != 'y':
                    break
    elif choice == "2":
        print("\nQuick training for demonstration (2500 episodes)...")
        agent = train_agent(episodes=2500)
        
        while True:
            play_against_human(agent)
            again = input("\nPlay again? (y/n): ")
            if again.lower() != 'y':
                break
    elif choice == "3":
        import os
        model_path = "c:\\GitHub\\TicTacToeRL\\models\\tictactoe_dqn.keras"
        
        if not os.path.exists(model_path):
            print(f"No trained model found at {model_path}")
            print("Please train a model first (option 1 or 2)")
            return
        
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
