import numpy as np
import jax.numpy as jnp
from scipy import stats


class Policy:
    def __init__(self, nState, nAction, temp, seed, epsilon=1e-6):
        self.nState = nState
        self.nAction = nAction
        self.temp = temp
        self.epsilon = epsilon
        self.rng = np.random.RandomState(int(seed))
        p_params = jnp.ones((self.nState * self.nAction)) * 1./self.nAction
        self.p_params = p_params

    def __call__(self, curr_state):
        p_params = self.p_params.reshape(self.nState, self.nAction)
        action_probs = p_params/np.sum(p_params, axis=1, keepdims=True)
        return self.rng.multinomial(1, action_probs[curr_state]).nonzero()[0][0] 
    
    def get_params(self):
        return self.p_params
    
    def update_params(self, p_params):
        self.p_params = np.clip(p_params, self.epsilon, 1. - self.epsilon)  # no deterministic
    
    def reset_params(self):
        self.p_params = jnp.ones((self.nState * self.nAction)) * 1./self.nAction

    def entropy(self, state):
        p_params = self.p_params.reshape(self.nState, self.nAction)
        action_probs = (p_params/np.sum(p_params, axis=1, keepdims=True))[state]
        return stats.entropy(action_probs)


def stable_softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator/denominator
    return softmax


class SoftmaxPolicy(Policy):
    def __call__(self, curr_state):
        p_params = self.p_params.reshape(self.nState, self.nAction)[curr_state]
        action_probs = stable_softmax(p_params)
        return self.rng.multinomial(1, action_probs).nonzero()[0][0]

    def update_params(self, p_params):
        self.p_params = p_params

    def entropy(self, state):
        p_params = self.p_params.reshape(self.nState, self.nAction)[state]
        action_probs = stable_softmax(p_params)
        return stats.entropy(action_probs)