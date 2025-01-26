import numpy as np


class QLearningAgent:
    def __init__(self):
        self.state_size = 2**11
        self.action_size = 3
        self.q_table = np.random.uniform(-1, 1, (self.state_size, self.action_size))
        self.alpha = 0.2
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.011
        self.epsilon_decay = 0.995

    def _state_to_index(self, state):
        return int("".join(map(str, state)), 2)

    def get_action(self, state):
        state_idx = self._state_to_index(state)
        if np.random.uniform() < self.epsilon:
            return np.random.randint(self.action_size)
        return np.argmax(self.q_table[state_idx])

    def update_q_table(self, state, action, reward, next_state):
        state_idx = self._state_to_index(state)
        next_state_idx = self._state_to_index(next_state)
        self.q_table[state_idx][action] += self.alpha * (
            reward + self.gamma * np.max(self.q_table[next_state_idx]) - self.q_table[state_idx][action]
        )
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)