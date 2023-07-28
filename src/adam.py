import jax.numpy as jnp


class Adam(object):
    def __init__(self, size, beta1=0.9, beta2=0.999, epsilon=1e-08):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = jnp.zeros(size, 'float32')
        self.v = jnp.zeros(size, 'float32')
        self.t = 0

    def update(self, globalg, stepsize=1e-4):
        self.t += 1
        a = stepsize * jnp.sqrt(1 - self.beta2**self.t)/(1 - self.beta1**self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = a * self.m / (jnp.sqrt(self.v) + self.epsilon)
        return step



