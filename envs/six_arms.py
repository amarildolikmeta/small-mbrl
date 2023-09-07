import numpy as np

def compute_probabilities(nS, nA):
    p = np.zeros((nS, nA, nS))
    # state 1
    p[0, 0, 1] = 1
    p[1, 0, 1] = p[1, 1, 1] = p[1, 2, 1] = p[1, 3, 1] = p[1, 5, 1] = 1
    p[1, 4, 0] = 1
    # state 2
    p[0, 1, 2] = 0.15
    p[0, 1, 0] = 0.85
    p[2, 0, 0] = p[2, 2, 0] = p[2, 3, 0] = p[2, 4, 0] = p[2, 5, 0] = 1
    p[2, 1, 2] = 1
    # state 3
    p[0, 2, 3] = 0.1
    p[0, 2, 0] = 0.9
    p[3, 2, 3] = 1
    p[3, 0, 0] = p[3, 1, 0] = p[3, 3, 0] = p[3, 4, 0] = p[3, 5, 0] = 1
    # state 4
    p[0, 3, 4] = 0.05
    p[0, 3, 0] = 0.95
    p[4, 3, 4] = 1
    p[4, 0, 0] = p[4, 1, 0] = p[4, 2, 0] = p[4, 4, 0] = p[4, 5, 0] = 1
    # state 5
    p[0, 4, 5] = 0.03
    p[0, 4, 0] = 0.97
    p[5, 4, 5] = 1
    p[5, 0, 0] = p[5, 1, 0] = p[5, 2, 0] = p[5, 3, 0] = p[5, 5, 0] = 1
    # state 6
    p[0, 5, 6] = 0.01
    p[0, 5, 0] = 0.99
    p[6, 5, 6] = 1
    p[6, 0, 0] = p[6, 1, 0] = p[6, 2, 0] = p[6, 3, 0] = p[6, 4, 0] = 1
    return p


def compute_rewards(nS, nA, rew):
    r = np.zeros((nS, nA, nS))
    r[1, 0, 1] = r[1, 1, 1] = r[1, 2, 1] = r[1, 3, 1] = r[1, 5, 1] = rew[0]
    for i in range(2, nS):
        r[i, i - 1, i] = rew[i - 1]
    return r


def compute_mu(nS):
    mu = np.zeros(nS)
    mu[0] = 1
    return mu

class SixArms:
    def __init__(self, gamma=0.99, seed=None):
        self.discount = self.gamma = gamma
        self.rng = np.random.RandomState(seed)
        nA = 6
        nS = 7
        self.nState = nS
        self.nAction = nA
        rew = [50, 133, 300, 800, 1660, 6000]
        self.initial_distribution = compute_mu(nS)
        self.P = compute_probabilities(nS=nS, nA=nA)
        self.R = np.sum(self.P * compute_rewards(nS, nA, rew), axis=2)
        self.reset()

    def get_name(self):
        return f'SixArms'

    def reset(self):
        self.state = np.random.choice(np.arange(self.P.shape[0]), p=self.initial_distribution)
        return self.state

    def reset_to_state(self, state):
        self.state = state
        return int(self.state)

    def get_state(self):
        return self.state

    def step(self, action):
        # assert list(range(self.nAction)).contains(action)
        next_state = np.random.choice(self.nState, p=self.P[self.state, action])
        if len(self.R.shape) == 3:
            reward = self.R[self.state, action, next_state]
        else:
            reward = self.R[self.state, action]
        self.state = next_state
        return next_state, reward, False, {}