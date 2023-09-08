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

        self.get_dynamics_and_rewards_distributions()
        self.P, r = self.get_mean_P_and_R()
        self.R = np.sum(self.P * r, axis=2)
        self.initial_distribution = np.zeros(self.N * 2 + 1)
        self.initial_distribution[0] = 1.
        self.reset()

    def get_dynamics_and_rewards_distributions(self):
        '''
            Implementation of the corresponding method from TabularEnvironment
        '''

        # Dict for transitions, P_probs[(s, a)] = [(s1, ...), (p1, ...)]
        P_probs = {}

        for n in range(self.N):
            for a in range(self.W):
                # Wide part transitions
                P_probs[(2 * n, a)] = [(2 * n + 1,), (1.00,)]

            P_probs[(2 * n + 1, 0)] = [(2 * n + 2,), (1.00,)]

        # Last state transitions to first state
        P_probs[(2 * self.N, 0)] = [(0,), (1.00,)]
        self.P_probs = P_probs

        def P(s, a):

            # Next states and transition probabilities
            s_, p = P_probs[(s, a)]

            # Sample s_ according to the transition probabilities
            s_ = np.random.choice(s_, p=p)

            return s_

        def R(s, a, s_):

            # Booleans for current and next state
            even_s, odd_s_ = s % 2 == 0, s_ % 2 == 1

            # Zero reward for transition from last to first state
            if s == 2 * self.N and s_ == 0:
                return 0.
            # High reward for correct action from odd state
            elif even_s and odd_s_ and (a == 0):
                return self.mu_h + self.sig_h * np.random.normal()
            # Low reward for incorrect action from odd state
            elif even_s and odd_s_:
                return self.mu_l + self.sig_l * np.random.normal()
            # Reward from even state
            else:
                return self.mu_n + self.sig_n * np.random.normal()

        return P, R


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

if __name__ == "__main__":
    env = WideNarrow()
    pi, V, Q = value_iteration(P=env.P, R=env.R, gamma=env.gamma, qs=True, max_iter=100000, tol=1e-10)
    print(pi)
    print(V)
    print(Q)
