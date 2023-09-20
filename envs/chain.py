import numpy as np
import sys
sys.path.append('..')
from src.utils import value_iteration


def compute_probabilities(slip, nS, nA):
    p = np.zeros((nS, nA, nS))
    for i in range(nS):
        p[i, 0, min(nS - 1, i + 1)] = 1 - slip
        for k in range(i + 1):
            p[i, 1, k] = (1 - slip) / (i + 1)
        # p[i, 1, max(0, i-1)] = 1-slip
        p[i, 1, 0] += slip
        p[i, 0, 0] = slip
    # p[0, 1, 0] = 1.
    # p[1, 1, 0] = (1. - slip) / 2 + slip
    return p


def compute_rewards(nS, nA, small, large):
    r = np.zeros((nS, nA, nS))
    for i in range(nS):
        r[i, 1, 0] = r[i, 0, 0] = small
    r[nS - 1, 0, nS - 1] = r[nS - 1, 1, nS - 1] = large
    return r


class Chain:
    """n-Chain environment
    This game presents moves along a linear chain of states, with two actions:
     0) forward, which moves along the chain but returns no reward
     1) backward, which returns to the beginning and has a small reward
    The end of the chain, however, presents a large reward, and by moving
    'forward' at the end of the chain this large reward can be repeated.
    At each action, there is a small probability that the agent 'slips' and the
    opposite transition is instead taken.
    The observed state is the current state in the chain (0 to n-1).
    This environment is described in section 6.1 of:
    A Bayesian Framework for Reinforcement Learning by Malcolm Strens (2000)
    http://ceit.aut.ac.ir/~shiry/lecture/machine-learning/papers/BRL-2000.pdf
    """
    def __init__(self, nState=5, prob_slip=0.2, discount=0.99, seed=None, small=2., large=10., uniform=False):
        self.prob_slip = prob_slip  # probability of 'slipping' an action
        self.small = small  # payout for 'backwards' action
        self.large = large  # payout at end of chain for 'forwards' action
        self.state = 0  # Start at beginning of the chain
        self.nAction = 2
        self.nState = nState
        self.discount = self.gamma = discount
        self.uniform = uniform
        if self.uniform:
            self.initial_distribution = np.ones(self.nState) / self.nState
        else:
            self.initial_distribution = np.zeros(self.nState)
            self.initial_distribution[0] = 1.
        self.rng = np.random.RandomState(seed)
        self.P = compute_probabilities(slip=prob_slip, nS=nState, nA=2)
        self.R = np.sum(self.P * compute_rewards(nS=nState, nA=2, small=small, large=large), axis=2)
        self.reset()

    def get_name(self):
        return f'Chain{self.nState}StateSlip{self.prob_slip}'

    def reset(self):
        self.state = np.random.choice(self.nState, p=self.initial_distribution)
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


if __name__ == "__main__":
    env = Chain()
    pi, V, Q = value_iteration(P=env.P, R=env.R, gamma=env.gamma, qs=True, max_iter=100000, tol=1e-10)
    print(pi)
    print(V)
    print(Q)
