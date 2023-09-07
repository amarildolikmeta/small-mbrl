import numpy as np

class WideNarrow:
    def __init__(self,  n=1, w=6, gamma=0.999, seed=None):
        self.discount = self.gamma = gamma
        self.N, self.W = n, w
        self.rng = np.random.RandomState(seed)
        nA = self.W
        nS = 2 * self.N + 1
        self.nState = nS
        self.nAction = nA
        self.mu_l, self.sig_l = 0., 1.
        self.mu_h, self.sig_h = 0.5, 1.
        self.mu_n, self.sig_n = 0, 1

        self.P, r = self.get_mean_P_and_R()
        np.sum(self.P * r, axis=2)
        self.initial_distribution = np.zeros(self.N * 2 + 1)
        self.initial_distribution[0] = 1.
        self.reset()

    def get_mean_P_and_R(self):

        P = np.zeros((2 * self.N + 1, self.W, 2 * self.N + 1))
        R = np.zeros((2 * self.N + 1, self.W, 2 * self.N + 1))

        for s in range(2 * self.N + 1):
            for a in range(self.W):

                # Uniform prob hack for dissallowed states
                if not ((s, a) in self.P_probs):
                    P[s, a, :] = 1. / (2 * self.N + 1)

                for s_ in range(2 * self.N + 1):

                    # Large negative reward for non-allowed transitions
                    R[s, a, s_] = -1e6

                    # Take action from last state
                    if s == 2 * self.N and s_ == 0 and a == 0:
                        P[s, a, s_] = 1.
                        R[s, a, s_] = 0.

                    # Take good action from even state
                    elif s % 2 == 0 and s_ == s + 1 and a == 0:
                        P[s, a, s_] = 1.
                        R[s, a, s_] = self.mu_h

                    # Take suboptimal action from even state
                    elif s % 2 == 0 and s_ == s + 1:
                        P[s, a, s_] = 1.
                        R[s, a, s_] = self.mu_l

                    # Take action from odd state
                    elif s % 2 == 1 and s_ == s + 1 and a == 0:
                        P[s, a, s_] = 1.
                        R[s, a, s_] = self.mu_n

        return P, R

    def get_name(self):
        return 'WideNarrow-N-{}_W-{}'.format(self.N, self.W)

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