import numpy as np
import jax.numpy as jnp
from scipy import stats
from jax import random


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


class LogisticPolicy(Policy):
    def __init__(self, nState, nAction, temp, seed, epsilon=1e-6):
        super().__init__(nState, nAction, temp, seed, epsilon=epsilon)
        self.reset_params()

    def __call__(self, curr_state):
        epsilon = jnp.diag(jnp.ones(2) * 1 / (1 + jnp.exp(-self.p_params)))
        action_probs = (1 - epsilon + epsilon.T)[curr_state]
        return self.rng.multinomial(1, action_probs).nonzero()[0][0]

    def reset_params(self):
        self.p_params = jnp.zeros(1)
    def update_params(self, p_params):
        self.p_params = p_params

    def entropy(self, state):
        epsilon = jnp.diag(jnp.ones(2) * 1 / (1 + jnp.exp(-self.p_params)))
        action_probs = (1 - epsilon + epsilon.T)[state]
        return stats.entropy(action_probs)


class LinearGaussianPolicy:
    def __init__(self, p_params=None, noise=None, dim=1, seed=0):
        if p_params is not None:
            self.p_params = p_params
            self.output, self.input = self.p_params.shape
        else:
            self.p_params = jnp.ones((dim, dim)) * np.random.rand((dim, dim))
        if noise is not None and isinstance(noise, (int,  float, complex)):
            noise = jnp.diag(jnp.ones(self.output) * noise)
        else:
            noise = jnp.diag(jnp.ones(self.output) * 0.1)
        self.noise = noise
        self.nState = self.input
        self.nAction = self.output
        self.key = random.PRNGKey(seed)

    def __call__(self, curr_state):
        return self.act(curr_state)

    def get_params(self):
        return self.p_params

    def update_params(self, p_params):
        self.p_params = p_params

    def reset_params(self):
        self.p_params = jnp.ones((self.nState * self.nAction)) * 1. / self.nAction

    def entropy(self, state):
        p_params = self.p_params.reshape(self.nState, self.nAction)
        action_probs = (p_params / np.sum(p_params, axis=1, keepdims=True))[state]
        return stats.entropy(action_probs)

    def set_weights(self, p_params, noise=None):
        self.p_params = p_params
        self.output, self.input = self.weights.shape
        if noise is not None and isinstance(noise, (int, float, complex)):
            noise = jnp.diag(np.ones(self.output) * noise)
        self.noise = noise

    def _add_noise(self):
        self.key, subkey = random.split(self.key)
        noise = random.multivariate_normal(subkey, jnp.zeros(self.output), self.noise, 1).T
        return noise

    def act(self, X, stochastic=True):
        y = jnp.dot(self.p_params, X)
        if self.noise is not None and stochastic:
            y += self._add_noise()
        return y

    def step(self, X, stochastic=False):
        return None, self.act(X, stochastic), None, None

    def compute_gradients(self, X, y, noise):
        mu = jnp.dot(self.p_params, X)
        return jnp.diag((jnp.dot(jnp.diag(1 / noise), jnp.dot((y - mu), X.T))))

def compute_gradients(p_params, X, y, noise):
    mu = jnp.dot(p_params, X)
    return jnp.diag((jnp.dot(jnp.diag(1 / noise), jnp.dot((y - mu), X.T))))

