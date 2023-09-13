import numpy as np
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import jax
from jax.nn import sigmoid
from itertools import product
from src.replay_buffer import ReplayBuffer
import scipy
import pdb
# from memory_profiler import profile

def get_id(args):
    """
    :param args:
    :return:
    """
    # TODO: Make these in a sensible order, so readable
    args_id = ''
    args_id += f'train_type{args.train_type}_'
    args_id += f'seed{args.seed}'
    args_id += f'risk_threshold{args.risk_threshold}_'
    args_id += f'policy_lr{args.policy_lr}_'
    args_id += f'num_samples_plan{args.num_samples_plan}_'
    # args_id += f'num_eps={args.num_eps}_'
    args_id += f'traj_len{args.traj_len}_'
    args_id += f'k_value{args.k_value}_'
    args_id += f'log_freq{args.log_freq}_'
    args_id += f'num_eps_eval{args.num_eps_eval}_'
    return args_id

def similar(L):
    return all(np.isclose(x, y, rtol=1e-1) for x,y in zip(L[-10:], L[-9:]))
    
def non_decreasing(L):
    return all((x<y) or np.isclose(x,y,rtol=1.) for x, y in zip(L, L[1:]))

def update_mle(nState, nAction, batch):
    transition_counts = np.zeros((nState, nAction, nState))
    rewards_estimate = np.zeros((nState, nAction))
    obses, actions, rewards, next_obses, not_dones, _ = batch
    for state, action, reward, next_state, not_done in zip(obses, actions, rewards, next_obses, not_dones):
        transition_counts[int(state), int(action), int(next_state)] += 1
        rewards_estimate[int(state), int(action)] = reward

    transition_counts = np.nan_to_num(transition_counts/np.sum(transition_counts, axis=2, keepdims=True))
    return transition_counts, rewards_estimate

def collect_sa(rng, env, nState, nAction, num_points_per_sa):
    replay_buffer = ReplayBuffer([1], [1], num_points_per_sa * (env.nState*env.nAction))
    count_sa = np.ones((env.nState, env.nAction)) * 0.#num_points_per_sa
    # init_states = init_distr.nonzero()
    # count_sa[init_states] = 0
    # print(env.nState, env.nAction)
    collected_all_sa = False if num_points_per_sa > 0 else True
    # while not collected_all_sa:
    done = False
    for state, action in product(range(env.nState), range(env.nAction)):
        while count_sa[state, action] < num_points_per_sa:
            env.reset_to_state(state)
            next_state, reward, done, _ = env.step(action)
            if count_sa[state, action] < num_points_per_sa:
                count_sa[state, action] += 1
                replay_buffer.add(state, action, reward, next_state, done, done)
            state = next_state
            # collected_all_sa = (count_sa >= num_points_per_sa).all()
            # print(count_sa)
    # print(count_sa)
    print('finished data collection')
    return replay_buffer


def collect_data(rng, env, nEps, nSteps, policy):
    replay_buffer = ReplayBuffer([1], [1], nEps * nSteps)
    for ep in range(nEps):
        # print(ep)
        done = False
        step = 0
        state = env.reset()
        while not done and step < nSteps:
        # for step in range(10):
            # action = rng.choice(action_space)
            if policy[state].shape == (): #optimal policy
                action = int(policy[state])
            else:
                action = rng.multinomial(1, policy[state]).nonzero()[0][0]
            next_state, reward, done, _ = env.step(action)
            replay_buffer.add(state, action, reward, next_state, done, done)
            state = next_state
            step += 1
    return replay_buffer


def discount(rewards, discount_factor) -> np.ndarray:
    rewards_np = np.asarray(rewards)
    t_steps = np.arange(len(rewards))
    r = rewards_np * discount_factor**t_steps
    r = r[::-1].cumsum()[::-1] / discount_factor**t_steps
    return r


def vmap_sigmoid(x):
    return jax.vmap(sigmoid, in_axes=0, out_axes=0)(x)


def get_log_policy(p_params, nState, nAction, temp):
    """
    :param p_params:
    :return:
    """
    p_params = p_params.reshape(nState, nAction)
    return jnp.log(p_params) - jnp.log(jnp.sum(p_params, axis=1, keepdims=True))


def stable_softmax(x):
    z = x - jnp.max(x, axis=1).T
    numerator = jnp.exp(z)
    denominator = jnp.sum(numerator, axis=1)
    softmax = numerator/denominator
    return softmax


def get_policy(p_params, nState, nAction):
    p_params = p_params.reshape(nState, nAction)
    policy = softmax(p_params)  # stable_softmax(p_params)
    return policy


def get_policy_logistic(p_params, nState, nAction):
    epsilon = jnp.diag(jnp.ones(2) * 1 / (1 + jnp.exp(-p_params)))
    antidiagonal_epsilon = jnp.fliplr(epsilon)
    diagonal_epsilon = (jnp.diag(jnp.ones(2)) - epsilon)
    action_probs = diagonal_epsilon + antidiagonal_epsilon
    policy = action_probs.reshape(nState, nAction)
    return policy

def project_onto_simplex(params, axis=1):
    zero = 1.e-6
    n_features = params.shape[axis]
    one = 1. - n_features * 1.e-6
    #sort descending
    u = jnp.flip(jnp.sort(params, axis=axis), axis=axis)
    cssv = jnp.cumsum(u, axis=axis)
    ind = np.arange(n_features) + 1
    ind = jnp.broadcast_to(ind, params.shape)
    cond = (u + (one - cssv) / ind) > zero
    masked_ind = jnp.where(cond, ind, -np.infty)
    rho = jnp.max(masked_ind, axis=axis).astype(int)
    rho = jnp.expand_dims(rho, axis=axis)
    cumsum_u_upto_rho = jnp.take_along_axis(cssv, rho - 1, axis=axis)
    lambda_ = (one - cumsum_u_upto_rho) / (rho * 1.0)
    w = jnp.maximum(params + lambda_, zero)
    return w


def log_softmax(vals, temp=1.):
    """Same function as softmax but not exp'd
        Args:
            vals : S x A. Applied row-wise
            temp (float, optional): Defaults to 1.. Temperature parameter
        Returns:
        """
    return (1. / temp) * vals - logsumexp((1. / temp) * vals, axis=1, keepdims=True)


def softmax(vals, temp=1.):
    """Batch softmax
    Args:
     vals : S x A. Applied row-wise
     temp (float, optional): Defaults to 1.. Temperature parameter
    Returns:
    """
    softmax = jnp.exp((1. / temp) * vals - logsumexp((1. / temp) * vals, axis=1, keepdims=True))
    return softmax


def bellmap_op(V, P, R, gamma):
    """
    Applies the optimal Bellman operator to a value function V.

    :param V: a value function. Shape: (S,)
    :return: the updated value function and the corresponding greedy action for each state. Shapes: (S,) and (S,)
    """
    S, A = P.shape[:2]
    Q = np.empty((S, A))
    if R.shape == (S, A, S):
        R = np.sum(P * R, axis=2)
    for s in range(S):
        Q[s] = R[s] + gamma * P[s].dot(V)

    return Q.argmax(axis=1), Q.max(axis=1), Q


def value_iteration(P, R, gamma, max_iter=1000, tol=1e-3, verbose=False, qs =False):
    """
    Applies value iteration to this MDP.

    :param max_iter: maximum number of iterations
    :param tol: tolerance required to converge
    :param verbose: whether to print info
    :return: the optimal policy and the optimal value function. Shapes: (S,) and (S,)
    """

    # Initialize the value function to zero
    V = np.zeros(P.shape[0],)

    for i in range(max_iter):
        # Apply the optimal Bellman operator to V
        pi, V_new, Q = bellmap_op(V, P, R, gamma)

        # Check whether the difference between the new and old values are below the given tolerance
        diff = np.max(np.abs(V - V_new))

        if verbose:
            print("Iter: {0}, ||V_new - V_old||: {1}, ||V_new - V*||: {2}".format(i, diff,
                                                                                  2*diff*gamma/(1-gamma)))

        # Terminate if the change is below tolerance
        if diff <= tol:
            break
        # Set the new value function
        V = V_new
    if qs:
        return pi, V, Q
    else:
        return pi, V


class Parameter:
    def __init__(self, value, min_value=None, max_value=None):
        self._initial_value = value
        self._min_value = min_value
        self._max_value = max_value
        self._n_updates = 0

    def __call__(self):
        return self.get_value()

    def get_value(self):
        new_value = self._compute()

        if self._min_value is None and self._max_value is None:
            return new_value
        else:
            return np.clip(new_value, self._min_value, self._max_value)

    def _compute(self, *idx, **kwargs):
        return self._initial_value

    def update(self):
        self._n_updates += 1


class LinearParameter(Parameter):
    """
    This class implements a linearly changing parameter according to the number
    of times it has been used.

    """
    def __init__(self, value, threshold_value, n):
        self._coeff = (threshold_value - value) / n

        if self._coeff >= 0:
            super().__init__(value, None, threshold_value)
        else:
            super().__init__(value, threshold_value, None)

    def _compute(self):
        return self._coeff * self._n_updates + self._initial_value


class ExponentialParameter(Parameter):
    """
    This class implements a exponentially changing parameter according to the
    number of times it has been used.

    """
    def __init__(self, value, exp=1., min_value=None, max_value=None):
        self._exp = exp

        super().__init__(value, min_value, max_value)

    def _compute(self):
        n = np.maximum(self._n_updates, 1)
        return self._initial_value / n ** self._exp


if __name__ == "__main__":
    rewards = [1, 2, 3, 4]
    discount_factor = 0.9
    answer = [1+2*0.9+3*(0.9**2)+4*(0.9**3), 2+3*0.9+4*(0.9**2), 3+4*0.9, 4]
    print(answer)
    print(discount(rewards, discount_factor))
    print(discount(rewards, discount_factor) == np.asarray(answer))