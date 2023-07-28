import numpy as np


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
    def __init__(self, nState, prob_slip, discount, seed):
        self.prob_slip = prob_slip  # probability of 'slipping' an action
        self.small = 2.  # payout for 'backwards' action
        self.large = 10.  # payout at end of chain for 'forwards' action
        self.state = 0  # Start at beginning of the chain
        self.nAction = 2
        self.nState = nState
        self.discount = discount
        self.initial_distribution = np.zeros(self.nState)
        self.initial_distribution[0] = 1.
        self.rng = np.random.RandomState(seed)

    def get_name(self):
        return f'Chain{self.nState}StateSlip{self.prob_slip}'

    def reset(self):
        self.state = 0
        return self.state

    def reset_to_state(self, state):
        self.state = state
        return int(self.state)

    def step(self, action):
        # assert list(range(self.nAction)).contains(action)
        if self.rng.rand() < self.prob_slip:
            action = not action  # agent slipped, reverse action taken
        if action:  # 'backwards': go back to the beginning, get small reward
            reward = self.small
            self.state = 0
        elif self.state < self.nState - 1:  # 'forwards': go up along the chain
            reward = 0
            self.state += 1
        else:  # 'forwards': stay at the end of the chain, collect large reward
            reward = self.large
        done = False
        return self.state, reward, done, {}

