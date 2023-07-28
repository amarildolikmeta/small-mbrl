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
    z = x - jnp.max(x, axis=1)
    numerator = jnp.exp(z)
    denominator = jnp.sum(numerator, axis=1)
    softmax = numerator/denominator
    return softmax


def get_policy(p_params, nState, nAction):
    p_params = p_params.reshape(nState, nAction)
    policy = stable_softmax(p_params)
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

if __name__ == "__main__":
    rewards = [1, 2, 3, 4]
    discount_factor = 0.9
    answer = [1+2*0.9+3*(0.9**2)+4*(0.9**3), 2+3*0.9+4*(0.9**2), 3+4*0.9, 4]
    print(answer)
    print(discount(rewards, discount_factor))
    print(discount(rewards, discount_factor) == np.asarray(answer))