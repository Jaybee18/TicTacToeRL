import keras
import tensorflow as tf
import numpy as np
from random import randint, random, sample
from collections import deque
import matplotlib.pyplot as plt

GAMMA = 0.99
epsilon = 1.0
EPSILON_MAX = 1.0
EPSILON_MIN = 0.01
LEARNING_RATE = 0.01
DECAY = 0.01

EPISODES = 1000

game_state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])

X = 1
O = -1

def print_game_state(state: list[int]):
    print("""
[{}, {}, {}]
[{}, {}, {}]
[{}, {}, {}]
""".format(*["X" if s == X else "O" if s == O else " " for s in state])
    )

def create_q_model():
    model = keras.Sequential([
        keras.layers.Dense(36, activation='relu', input_shape=(9,)),
        keras.layers.Dense(36, activation='relu'),
        keras.layers.Dense(9, activation='linear'),
        # keras.layers.Softmax() ChatGPT sagt das ist falsch für q learning
    ])
    model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), metrics=['accuracy'])
    return model

def has_won(state: list[int], player: int) -> bool | None:
    """Checks if player has won or lost returning True or False. Returns None if draw."""
    winning_combinations = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8],
        [0, 4, 8],
        [2, 4, 6],
    ]
    for combo in winning_combinations:
        if state[combo[0]] == state[combo[1]] == state[combo[2]] != 0:
            return state[combo[0]] == player
    if 0 not in state:
        return None
    return False

def is_terminated(state: list[int]) -> bool:
    return 0 not in state or has_won(state, X) or has_won(state, O)

def reset_game_state() -> list[int]:
    return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])

def reward_for_move(before: list[int], after: list[int], player: int) -> float:
    if not is_terminated(after):
        return 0.0
    
    won = has_won(after, player)
    if won:
        logarithmic_speed_bonus = np.log1p((after == 0).sum()) * 2
        return 10.0 + logarithmic_speed_bonus
    
    if won is None:
        return 0.0
    
    if not won:
        return -10.0

def train(replay_memory, model: keras.Model, target_model, done):
    discount_factor = 0.9

    MIN_REPLAY_SIZE = 100
    if len(replay_memory) < MIN_REPLAY_SIZE:
        return

    batch_size = min(64, len(replay_memory))
    mini_batch = sample(replay_memory, batch_size)
    current_states = np.array([transition[0] for transition in mini_batch])
    current_qs_list = model.predict(current_states, verbose=False)
    new_current_states = np.array([transition[3] for transition in mini_batch])
    future_qs_list = target_model.predict(new_current_states, verbose=False)

    X = []
    Y = []
    for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
        if not done:
            max_future_q = reward + discount_factor * np.max(future_qs_list[index])
        else:
            max_future_q = reward

        # current_qs = current_qs_list[index]
        # current_qs[action] = (1 - LEARNING_RATE) * current_qs[action] + LEARNING_RATE * max_future_q
        current_qs = current_qs_list[index].copy()
        current_qs[action] = max_future_q  # standard TD target assignment

        X.append(observation)
        Y.append(current_qs)
    model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=False, shuffle=True)

def get_random_valid_move(state: list[int], player: int) -> int:
    """Find a random valid move and return it."""
    max_tries = 10000
    for _ in range(max_tries):
        new_state = state.copy()
        action = randint(0, 8)
        new_state[action] = player
        if valid_move(state, new_state):
            return action
    raise Exception

def valid_move(before: np.ndarray, after: np.ndarray) -> bool:
    return (before == 0).sum() - (after == 0).sum() == 1

def main():
    global game_state, epsilon
    print("Hello from tictactoerl!")

    model = create_q_model()
    model_target = create_q_model()
    model_target.set_weights(model.get_weights()) # Synchronize the models

    # Experience replay buffers
    replay_memory = deque(maxlen=50_000)

    # Track rewards for learning visualization
    rewards_history = []

    # Training
    for episode in range(EPISODES):
        print(f"Episode {episode}")
        total_rewards_for_episode = 0

        game_state = reset_game_state()

        # Random Mover (O) begins
        game_state[get_random_valid_move(game_state, O)] = O
        print_game_state(game_state)

        while True:
            # Explore or determine next move
            action = None
            if random() < epsilon:
                action = get_random_valid_move(game_state, X)
            else:
                prediction: np.ndarray = model.predict(np.expand_dims(game_state, 0), verbose=False)

                # Choose the action with the highest Q-value that is also a valid move
                qvals = prediction[0].copy()
                # mask invalid actions so they are never selected
                invalid_mask = (game_state != 0)
                qvals[invalid_mask] = -np.inf
                action = int(np.argmax(qvals))
            
            # Execute the move, then execute the enemies response
            # and observe the new game state
            new_game_state = game_state.copy()
            new_game_state[action] = X
            new_game_state[get_random_valid_move(new_game_state, O)] = O
            
            print_game_state(new_game_state)

            # Determine reward for move
            reward = reward_for_move(game_state, new_game_state, X)

            # Determine if the game is finished
            done = is_terminated(new_game_state)

            # Remember turn for learning
            replay_memory.append([game_state, action, reward, new_game_state, done])

            # Update the main network
            train(
                replay_memory,
                model,
                model_target,
                done
            )

            total_rewards_for_episode += reward
            game_state = new_game_state

            if done:
                if episode % 30 == 0:
                    model_target.set_weights(model.get_weights())
                rewards_history.append(total_rewards_for_episode)
                break

        epsilon = EPSILON_MIN + (EPSILON_MAX - EPSILON_MIN) * np.exp(-DECAY * episode)

        print(f"Total rewards for episode: {total_rewards_for_episode}")
    
    plt.plot(rewards_history)
    plt.show()

if __name__ == "__main__":
    main()
