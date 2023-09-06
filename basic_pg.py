import copy
import os
from typing import Tuple
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from envs.doubleloop import DoubleLoop
from envs.chain import Chain
from src.policy import Policy, SoftmaxPolicy, LogisticPolicy
from src.utils import get_policy, get_policy_logistic
from src.utils import value_iteration


def policy_evaluation(mdp: Tuple[np.ndarray], policy: jnp.ndarray, nState, discount):
    # Vals[state, timestep]
    r_matrix, p_matrix = mdp
    ppi = jnp.einsum('sat,sa->st', p_matrix, policy)
    rpi = jnp.einsum('sa,sa->s', r_matrix, policy)
    v_pi = jnp.linalg.solve(np.eye(nState) -
                            discount*ppi, rpi)
    return v_pi


def policy_performance(r, p, policy_params, initial_distribution, nState, nAction, discount):
    policy = get_policy(policy_params, nState, nAction)
    vf = policy_evaluation((r, p), policy, nState, discount)
    return initial_distribution @ vf


def policy_performance_logistic(r, p, policy_params, initial_distribution, nState, nAction, discount):
    policy = get_policy_logistic(policy_params, nState, nAction)
    vf = policy_evaluation((r, p), policy, nState, discount)
    return initial_distribution @ vf



def pg(R, P, initial_distribution, gamma=0.99, max_iters=100, period=10, tolerance=1e-4, verbose=False):
    nState = P[0].shape[0]
    nAction = P[0].shape[1]
    performances = []
    grads = []
    grad_norms = []
    converged = False
    it = 0
    grad_perf = jax.value_and_grad(policy_performance, 2)
    vmap_grad = jax.vmap(grad_perf, in_axes=(0, 0, None, None, None, None, None))
    while not converged and it < max_iters:
        params = policy.get_params()
        perf, U_pi_j_grad = jax.lax.stop_gradient(vmap_grad(R, P, params, initial_distribution,
                                                            nState, nAction, gamma))
        grad_norm = jnp.linalg.norm(U_pi_j_grad[0])
        converged = grad_norm < tolerance
        performances.append(perf[0])
        grads.append(U_pi_j_grad[0])
        grad_norms.append(grad_norm)
        new_params = params + learning_rate * U_pi_j_grad[0]
        policy.update_params(new_params)
        if verbose:
            if it % period == 0:
                print("Finished Grad step:" + str(it + 1))
        it += 1
    return performances, grads, grad_norms

if __name__ == "__main__":

    seed = np.random.randint(low=0, high=1000000)
    gamma = 0.99
    environment = "loop"
    use_softmax = True
    use_logistic = False
    max_iters = 100
    period = 10
    learning_rate = 1
    tolerance = 1e-3
    temp = 1.0
    save_dir = "outputs/pg/" + environment
    verbose = True

    if environment == "loop":
            env = DoubleLoop(seed=seed, gamma=gamma)
    elif environment == "chain":
            env = Chain(discount=gamma, seed=seed)
    else:
            raise ValueError("Env not implemented:" + environment)

    nState = env.nState
    nAction = env.nAction

    if use_softmax:
            policy = SoftmaxPolicy(nState, nAction, temp, seed)
            policy_performance = policy_performance
    elif use_logistic:
            policy = LogisticPolicy(nState, nAction, temp, seed)
            policy_performance = policy_performance_logistic
    else:
            policy = Policy(nState, nAction, temp, seed)
            policy_performance = policy_performance


    R = np.array([env.R])
    P = np.array([env.P])
    initial_distribution = env.initial_distribution

    _, V = value_iteration(P=P[0], R=R[0], gamma=gamma, max_iter=100000, tol=1e-10,
                                       qs=False)
    optimal_performance = initial_distribution @ V

    performances, grads, grad_norms = pg(R=R, P=P, initial_distribution=initial_distribution, gamma=gamma,
                                         max_iters=max_iters, tolerance=tolerance, verbose=verbose, period=period)

    n_rows = 1
    n_cols = 2
    # save results
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(save_dir + "/policy_performance.npy", performances)
    np.save(save_dir + "/grads.npy", grads)
    np.save(save_dir + "/grad_norms.npy", grad_norms)

    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 15))
    x = np.array([x for x in range(len(performances))])
    axs[0].plot(x, performances, label=" policy return", c="b")
    axs[0].plot([x[0], x[-1]], [optimal_performance, optimal_performance], label="optimal", c="black", linestyle='--')
    axs[1].plot(x, grad_norms, label="gradient norm", c="b")
    axs[0].set_xlabel("grad step")
    axs[1].set_xlabel("grad step")
    axs[0].set_ylabel("Return")
    axs[1].set_ylabel("Gradient Norm")
    axs[0].legend()
    plt.show()
    fig.savefig(save_dir + '/pg_curve.pdf')
    fig.savefig(save_dir + '/pg_curve.png')