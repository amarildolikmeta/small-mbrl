import numpy as np
import sys
sys.path.append('..')
from src.utils import value_iteration


class DoubleLoop:
    def __init__(self, seed=None, gamma=0.99, uniform=False) -> None:
        self.nState = 9 
        self.nAction = 2
        self.gamma = gamma
        self.uniform = uniform
        if self.uniform:
            self.initial_distribution = np.ones(self.nState) / self.nState
        else:
            self.initial_distribution = np.zeros(self.nState)
            self.initial_distribution[0] = 1.
        self.state = 0   
        self.rng = np.random.RandomState(seed)
        self.P = np.zeros((9, 2, 9))
        self.R = np.zeros((9, 2))
        self.P[0, 0, 1] = 1.
        self.P[0, 1, 5] = 1.
        for i in [1, 2, 3]:
            self.P[i, :, i + 1] = 1.
        self.P[4, :, 0] = 1.
        self.R[4, :] = 1.
        for i in [5, 6, 7]:
            self.P[i, 0, 0] = 1.
            self.P[i, 1, i + 1] = 1.
        self.P[8, :, 0] = 1.
        self.R[8, :] = 2.
        self.reset()

    def reset(self):
        self.state = np.random.choice(self.nState, p=self.initial_distribution)
        return self.state

    def step(self, action):
        if self.state == 0:
            if action == 0:
                next_state = 1
                reward = 0
            else:
                next_state = 5
                reward = 0
        elif self.state in [1, 2, 3]:
            next_state = self.state + 1
            reward = 0
        elif self.state == 4:
            next_state = 0
            reward = 1
        elif self.state in [5, 6, 7]:
            if action == 0:
                next_state = 0
                reward = 0
            elif action == 1:
                next_state = self.state + 1 
                reward = 0
        elif self.state == 8:
            next_state = 0
            reward = 2
        else:
            raise ValueError(f'State {self.state} is not defined')
        self.state = next_state
        return next_state, reward, False, {}

    def get_state(self):
        return self.state

    def get_name(self):
        return "DoubleLoop"


if __name__ == "__main__":
    env = DoubleLoop(gamma=0.99)

    pi, V, Q = value_iteration(P=env.P, R=env.R, gamma=env.gamma, qs=True, max_iter=100000, tol=1e-10)
    print(pi)
    print(V)
    print(Q)
    policy = np.array([[1, 0], [0, 1]])
    ppi = np.einsum('sat,sa->st', env.P, policy)
    rpi = np.einsum('sa,sa->s', env.R, policy)
    v_pi = np.linalg.solve(np.eye(2) -
                            env.gamma * ppi, rpi)
    print(v_pi)