import numpy as np
import random
from keras.activations import relu, linear
import gc
import keras

class DeepQLearning:

    def __init__(self, env, gamma, epsilon, epsilon_min, epsilon_dec, episodes, batch_size, memory, model, max_steps, target_model=None, update_target_every=10):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.episodes = episodes
        self.batch_size = batch_size
        self.memory = memory
        self.model = model
        self.max_steps = max_steps
        # Use the provided target_model, or default to the online model if none is provided.
        self.target_model = target_model if target_model is not None else model
        # Frequency (in episodes) at which the target network is updated.
        self.update_target_every = update_target_every

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.env.action_space.n)
        action = self.model.predict(state, verbose=0)
        return np.argmax(action[0])

    def experience(self, state, action, reward, next_state, terminal):
        self.memory.append((state, action, reward, next_state, terminal)) 

    def experience_replay(self):
        if len(self.memory) > self.batch_size:
            batch = random.sample(self.memory, self.batch_size)
            states = np.array([i[0] for i in batch])
            actions = np.array([i[1] for i in batch])
            rewards = np.array([i[2] for i in batch])
            next_states = np.array([i[3] for i in batch])
            terminals = np.array([i[4] for i in batch])
            states = np.squeeze(states)
            next_states = np.squeeze(next_states)

            # Use the target network to get the next Q-values.
            next_q_values = self.target_model.predict_on_batch(next_states)
            next_max = np.amax(next_q_values, axis=1)

            targets = rewards + self.gamma * next_max * (1 - terminals)
            targets_full = self.model.predict_on_batch(states)
            indexes = np.arange(self.batch_size)
            targets_full[indexes, actions] = targets

            self.model.fit(states, targets_full, epochs=1, verbose=0)

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_dec

    def train(self):
        rewards = []
        for i in range(self.episodes + 1):
            (state, _) = self.env.reset()
            state = np.reshape(state, (1, self.env.observation_space.shape[0]))
            score = 0
            steps = 0
            done = False

            while not done:
                steps += 1
                action = self.select_action(state)
                next_state, reward, terminal, truncated, _ = self.env.step(action)
                if terminal or truncated or (steps > self.max_steps):
                    done = True
                score += reward
                next_state = np.reshape(next_state, (1, self.env.observation_space.shape[0]))
                self.experience(state, action, reward, next_state, terminal)
                state = next_state
                self.experience_replay()

                if done:
                    print(f'Episode: {i+1}/{self.episodes}. Score: {score}')
                    break

            rewards.append(score)

            # Update the target network every `update_target_every` episodes.
            if i % self.update_target_every == 0:
                self.target_model.set_weights(self.model.get_weights())

            gc.collect()

        keras.backend.clear_session()
        return rewards
