import numpy as np
from numpy.random import RandomState

from envs.mdp import MDP


class continuousLQR(object):
    def __init__(self):
        super(continuousLQR, self).__init__()
        self.seed()
        self.A = onp.array([[0.9, 0.4], [-0.4, 0.9]])
        self.B = onp.eye(self.A.shape[0]) * 0.1
        self.states_dim = self.A.shape[0]
        self.actions_dim = self.B.shape[1]
        self.state = self.reset()
        self.discount_factor = 0.9
        
    def seed(self, seed=None):
        onp.random.seed(seed)

    def reset(self):
        self.state = -1*onp.ones((self.states_dim,))
        return self.state

    def step(self, action, state, key):
        del key
        reward = -(onp.dot(state.T, state) + onp.dot(action.T, action))
        state_prime = self.A.dot(state) + self.B.dot(action) + onp.random.normal(0, 0.1, 2)

        self.state = state_prime
        return state_prime, reward, 0, {}


def Garnet(StateSize = 5, ActionSize = 2, GarnetParam = (1,1)):
    # Initialize the transition probability matrix
    P = [np.matrix(np.zeros((StateSize, StateSize))) for act in
         range(ActionSize)]

    R = np.zeros((StateSize, 1))

    b_P = GarnetParam[0] # branching factor
    b_R = GarnetParam[1] # number of non-zero rewards

    # Setting up the transition probability matrix
    for act in range(ActionSize):
        for ind in range(StateSize):
            pVec = np.zeros(StateSize) + 1e-6
            p_vec = np.append(np.random.uniform(0,1,b_P - 1),[0,1])
            p_vec = np.diff(np.sort(p_vec))
            pVec[np.random.choice(StateSize, b_P, replace=False)] = p_vec
            pVec /= sum(pVec)
            P[act][ind,:] = pVec

    R[np.random.choice(StateSize, b_R, replace=False)] = np.random.uniform(0,1,b_R)[:, np.newaxis]

    return np.array(P), np.tile(np.array(R), ActionSize), 0.9


class GarnetMDP:
    def __init__(self, states_dim, actions_dim, garnet_param=(1,1)):
        self.seed()
        self.nState = states_dim
        self.nAction = actions_dim
        self.P, self.R, self.discount = Garnet(states_dim,
                                               actions_dim,
                                               garnet_param)
        self.P = self.P.reshape(states_dim, actions_dim, states_dim)
        self.initial_distribution = np.ones(self.nState)*1.0/self.nState
        self.garnet_param = garnet_param

    def get_name(self):
        return f'Garnet({self.nState},{self.nAction},{self.garnet_param})'

    def seed(self, seed=None):
        self.rng = RandomState(seed)

    def reset(self):
        self.state = self.rng.multinomial(1, self.initial_distribution).nonzero()[0][0]
        return self.state

    def step(self, action):
        reward = self.R[self.state, action]
        state_prime = self.rng.multinomial(1, self.P[self.state, action]).nonzero()[0][0]
        self.state = state_prime
        return state_prime, reward, 0, {}


class State2MDP:
    ''' Class for linear quadratic dynamics '''

    def __init__(self, SEED):
        '''
        Initializes:
        tuple (P, R, gamma) where the first element is a tensor of shape
        (A x S x S), the second element 'R' has shape (S x A) and the
        last element is the scalar (float) discount factor.
        '''
        self.P = np.array([[[0.7, 0.3], [0.2, 0.8]],
                            [[0.99, 0.01], [0.99, 0.01]]
                           ])
        # self.p = onp.array([[[1.-1e-6, 1e-6], [1e-6, 1.-1e-6]],
        #                     [[1.-1e-6, 1e-6], [1.-1e-6, 1e-6]]
        #                    ])
        self.R = np.array(([[-0.45, -0.1],
                                [0.5, 0.5]
                            ]))
        self.discount = 0.9
        self.state = None
        self.nAction, self.nState = self.P.shape[:2]
        self.initial_distribution = np.ones(self.nState) / self.nState
        self.rng = np.random.RandomState(SEED)
        
    def get_name(self):
        return 'State2MDP'
        
    # def seed(self, key):
    #     '''set random seed for environment'''
    #     # onp.random.seed(key)
    #     self.rng = np.random.RandomState(key)
    #     # self.key = key#jax.random.PRNGKey(key)

    def reset(self):
        all_states = np.arange(self.nState)
        # self.key, subkey = jax.random.split(self.key)
        self.state = all_states[self.rng.multinomial(1, self.initial_distribution).nonzero()] #jax.random.categorical(subkey,
        # self.initial_distribution) #

        # self.state = jax.random.categorical(key, self.initial_distribution)
        # return onp.asarray(self.state)
        return self.state[0]

    def step(self, action):#, key):
        all_states = np.arange(self.nState)
        next_state_probs = self.P[action, self.state]
        # # next_state = onp.asarray([int(jax.random.bernoulli(key, p=next_state_probs[0][0]))])
        
        # self.key, subkey = jax.random.split(self.key)
        # next_state = jax.random.categorical(subkey, self.p[action, self.state])
        # next_state = jax.random.categorical(key, self.p[action, state])
        next_state = all_states[self.rng.multinomial(1, next_state_probs[0]).nonzero()]
        reward = self.R[self.state, action][0]

        terminal = False
        return (next_state[0], reward, terminal, {})


class SimpleMDP(MDP):
    def __init__(self, epsilon=0.05, gamma=0.99, seed=123456):
        assert -0.5 <= epsilon <= 0.5
        P = np.zeros((2, 2, 2))
        P[0, 0, 0] = 0.5 - epsilon
        P[0, 0, 1] = 0.5 + epsilon

        P[0, 1, 0] = 0.5 + epsilon
        P[0, 1, 1] = 0.5 - epsilon

        P[1, 0, 0] = 1.
        P[1, 0, 1] = 0

        P[1, 1, 0] = 1.
        P[1, 1, 1] = 0

        R = np.zeros((2, 2))
        R[1, 1] = 1.

        mu = np.zeros(2)
        mu[0] = 1.
        super().__init__(n_states=2, n_actions=2, transitions=P, rewards=R, two_d_r=True, init_state=mu, gamma=gamma)
        self.rng = np.random.RandomState(seed)
        self.initial_distribution = mu
        self.nState = 2
        self.nAction = 2
        self.discount = gamma

    def get_name(self):
        return '2StateMDP'


class State_3_MDP(object):
    def __init__(self):
        '''
        Initializes:
        tuple (P, R, gamma) where the first element is a tensor of shape
        (A x S x S), the second element 'R' has shape (S x A) and the
        last element is the scalar (float) discount factor.
        '''
        super(State_3_MDP, self).__init__()
        self.p = onp.array([[[0.6, 0.4 - 1e-6, 1e-6], [0.1, 0.8, 0.1], [0.9 - 1e-6, 1e-6, 0.1]],
                  [[0.98, 0.01, 0.01], [0.2, 1e-6, 0.8 - 1e-6], [1e-6, 0.3, 0.7 - 1e-6]]
                  ])
        self.r = onp.array(([[0.1, -0.15],
                   [0.1, 0.8],
                   [-0.2, -0.1]
                   ]))
        self.discount_factor = 0.9
        self.state = None
        self.n_actions, self.n_states = self.p.shape[:2]
        self.initial_distribution = onp.ones(self.n_states) / self.n_states

    def seed(self, key):
        '''set random seed for environment'''
        onp.random.seed(key)
        # self.key = key

    def reset(self):
        all_states = onp.arange(self.n_states)
        self.state = all_states[onp.random.multinomial(1, self.initial_distribution).nonzero()]
        return onp.asarray(self.state)

    def step(self, action):
        all_states = onp.arange(self.n_states)
        next_state_probs = self.p[action, self.state]
        # next_state = onp.asarray([int(jax.random.bernoulli(key, p=next_state_probs[0][0]))])
        next_state = all_states[onp.random.multinomial(1, next_state_probs[0]).nonzero()]
        reward = self.r[self.state, action]

        terminal = False
        return (next_state, reward, terminal, {})

