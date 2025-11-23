import keras
from random import random, sample
from collections import deque
import numpy as np
from agent import TicTacToeAgent


def _create_q_model():
    return keras.Sequential(
        [
            keras.layers.Dense(64, activation="relu", input_shape=(9,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(9, activation="linear"),
        ]
    )

class DQNAgent(TicTacToeAgent):
    def __init__(self, player: int):
        super().__init__(player)
        self.opponent = -player
        self.model = _create_q_model()
        self.target_model = _create_q_model()
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
