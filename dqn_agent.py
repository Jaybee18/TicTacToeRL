import os
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
    def __init__(self, player: int, learning_rate: float = 0.001, gamma: float = 0.95, 
                 epsilon: float = 1.0, epsilon_min: float = 0.01, epsilon_decay: float = 0.995,
                 batch_size: int = 32, memory_size: int = 10000):
        super().__init__(player)
        self.opponent = -player
        
        # Store hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.episodes_trained = 0
        
        self.model = _create_q_model()
        self.target_model = _create_q_model()
        self.target_model.set_weights(self.model.get_weights())
        
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss_fn = keras.losses.MeanSquaredError()

        self.model.compile(optimizer=self.optimizer, loss=self.loss_fn)
        self.target_model.compile(optimizer=self.optimizer, loss=self.loss_fn)
        
        self.memory = deque(maxlen=self.memory_size)
        self.batch_size = batch_size
    
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
        
        # Update Q values with smooth blending to prevent catastrophic forgetting
        for i in range(self.batch_size):
            if dones[i]:
                target = rewards[i]
            else:
                target = rewards[i] + self.gamma * np.max(next_q[i])
            
            # Blend old and new Q-values (0.1 old + 0.9 new)
            current_q[i][actions[i]] = current_q[i][actions[i]] * 0.1 + target * 0.9
        
        # Train model
        self.model.fit(states, current_q, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_model(self):
        """Copy weights from model to target model"""
        self.target_model.set_weights(self.model.get_weights())
    
    def _generate_filename(self) -> str:
        """Generate filename based on hyperparameters."""
        # Format floating point numbers to use comma as decimal separator for filename
        lr_str = str(self.learning_rate).replace('.', ',')
        gamma_str = str(self.gamma).replace('.', ',')
        eps_min_str = str(self.epsilon_min).replace('.', ',')
        eps_decay_str = str(self.epsilon_decay).replace('.', ',')
        
        filename = f"model_episodes_{self.episodes_trained}_lr_{lr_str}_gamma_{gamma_str}_eps_min_{eps_min_str}_eps_decay_{eps_decay_str}_batch_{self.batch_size}_mem_{self.memory_size}.keras"
        return filename
    
    def save(self, filepath: str = None):
        """Save the model weights to a file."""
        if filepath is None:
            filename = self._generate_filename()
            filepath = os.path.join("c:\\GitHub\\TicTacToeRL\\models", filename)
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
        return filepath
    
    def load(self, filepath: str):
        """Load the model weights from a file."""
        self.model = keras.models.load_model(filepath)
        self.target_model.set_weights(self.model.get_weights())
        print(f"Model loaded from {filepath}")
