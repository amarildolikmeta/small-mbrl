import os
import argparse
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import jax
from envs.doubleloop import DoubleLoop
from envs.chain import Chain
from envs.six_arms import SixArms
from envs.three_arms import TreeArms
from envs.wide_narrow import WideNarrow
from envs.basic_envs import SimpleMDP
from envs.gridworld import GridWorld
from src.policy import Policy, SoftmaxPolicy, LogisticPolicy
from src.utils import value_iteration, Parameter, LinearParameter, ExponentialParameter
from src.model import DirichletModel
from basic_pg import policy_performance
import seaborn as sns
from matplotlib.lines import Line2D


def collect_samples_and_update_prior(agent, env, num_samples):
    for j in range(num_samples):
        state = env.get_state()
        action = jax.lax.stop_gradient(agent.policy(state))
        next_state, reward, done, _ = env.step(action)
        agent.update_obs(state, action, reward, next_state, done)
        if done:
            env.reset()


def compute_statistics(values, objective_type, regularization, alpha, delta, grads=None):
    values = np.asarray(values)
    L_pi = values.shape[0]
    sorted_values = np.sort(values)
    if grads is not None:
        grads = np.asarray(grads)
        sorted_grads = [x for _, x in sorted(zip(values, grads), key=lambda x: x[0])]

    avg_performance = np.mean(values, axis=0)
    floor_index = int(np.floor(alpha * L_pi))
    ceal_index = int(np.ceil(delta * L_pi))
    var_alpha = sorted_values[floor_index]
    var_delta = sorted_values[ceal_index]
    cvar_alpha = np.mean(sorted_values[:floor_index], axis=0)
    cvar_delta = np.mean(sorted_values[ceal_index:], axis=0)
    upper_bound = sorted_values[-1]
    lower_bound = sorted_values[0]
    if objective_type == "max":
        objective = sorted_values[-1]
        if grads is not None:
            grad = sorted_grads[-1]
    elif objective_type == "pg":
        objective = avg_performance
        if grads is not None:
            grad = np.mean(grads, axis=0)
    elif objective_type == "upper_cvar":
        objective = cvar_delta
        if grads is not None:
            grad = np.mean(sorted_grads[ceal_index:], axis=0)
    elif objective_type == "upper_delta":
        objective = var_delta
        if grads is not None:
            grad = sorted_grads[ceal_index]
    else:
        raise ValueError("Objective not implemented:" + objective_type)
    if regularization == "lower_bound":
        regularization_term = var_alpha
        if grads is not None:
            constraint_grad = sorted_grads[floor_index]
    elif regularization == "cvar":
        regularization_term = cvar_alpha
        if grads is not None:
            constraint_grad = np.mean(sorted_grads[:floor_index], axis=0)
    else:
        raise ValueError("Regularization not implemented:" + regularization)

    if grads is None:
        grad = None
        constraint_grad = None
    return avg_performance, cvar_alpha, cvar_delta, var_alpha, var_delta, upper_bound, lower_bound, objective, regularization_term, grad,\
        constraint_grad


def policy_optimization(agent, policy_performance, num_posterior_samples=100, gamma=0.99, objective_type="max",
                        regularization="cvar", alpha=0.1, delta=0.9, lambda_=0.,  optimization_iterations=100,
                        optimization_tolerance=1e-4, policy_lr=1., reset_policy=False, resample=False):
    p_params = agent.policy.get_params()
    R_j, P_j = agent.multiple_sample_mdp(num_posterior_samples)
    init_dist = agent.initial_distribution
    nStates, nAction = P_j[0].shape[:2]
    grad_perf = jax.value_and_grad(policy_performance, 2)
    vmap_grad = jax.vmap(grad_perf, in_axes=(0, 0, None, None, None, None, None))
    U_pi_j, U_pi_j_grad = jax.lax.stop_gradient(vmap_grad(R_j, P_j, p_params, init_dist,
                                                          nState, nAction, gamma))
    initial_distribution = U_pi_j
    it = 0
    if reset_policy:
        agent.policy.reset_params()
    converged = False
    while it < optimization_iterations and not converged:
        p_params = agent.policy.get_params()
        U_pi_j, U_pi_j_grad = jax.lax.stop_gradient(vmap_grad(R_j, P_j, p_params, init_dist,
                                                              nState, nAction, gamma))
        U_pi = np.asarray(U_pi_j)
        U_pi_grads = np.asarray(U_pi_j_grad)

        avg_performance, cvar_alpha, cvar_delta, var_alpha, var_delta, upper_bound, lower_bound, objective,\
            regularization_term, grad, constraint_grad = compute_statistics(values=U_pi, objective_type=objective_type,
                                                                            regularization=regularization,
                                                                            grads=U_pi_grads,
                                                                            alpha=alpha,
                                                                            delta=delta)

        objective += lambda_ * regularization_term
        p_grad = grad + lambda_ * constraint_grad
        p_params = p_params + policy_lr * p_grad
        grad_norm = np.linalg.norm(p_grad)
        converged = grad_norm < optimization_tolerance
        agent.policy.update_params(p_params)
        it += 1
        if resample:
            R_j, P_j = agent.multiple_sample_mdp(num_posterior_samples)
    p_params = agent.policy.get_params()
    U_pi_j, U_pi_j_grad = jax.lax.stop_gradient(vmap_grad(R_j, P_j, p_params, init_dist,
                                                          nState, nAction, gamma))
    return lower_bound, var_alpha, cvar_alpha, avg_performance, cvar_delta, var_delta, upper_bound, grad_norm,\
        converged, it, U_pi_j, initial_distribution, objective


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--environment', type=str, default='loop', choices=["loop", "chain", "widenarrow", "sixarms",
                                                                            "threearms", "2_state", "grid"])
    parser.add_argument('--base_dir', type=str, default='')
    parser.add_argument('--objective', type=str, default='max', choices=["max", "upper_cvar", "upper_delta", "pg"],
                        help="Choice of objective function")
    parser.add_argument('--regularization', type=str, default='cvar', choices=["cvar", "lower_bound"],
                        help="Choice of regularizer")
    parser.add_argument('--alpha', type=float, default=0.1, help="Risk ratio")
    parser.add_argument('--delta', type=float, default=0.9, help="Upper confidence")
    parser.add_argument('--use_logistic', action="store_true", help="Use logistic policy in 2 state mdp")
    parser.add_argument('--use_softmax', action="store_true", help="Use softmax policy")
    parser.add_argument('--resample', action="store_true", help="Resample from prior at each optimization")
    parser.add_argument('--reset_policy', action="store_true", help="Reset policy randomly after prior changes")
    parser.add_argument('--max_iters', type=int, default=100, help="Iteration of the inner loop")
    parser.add_argument('--iterations', type=int, default=100, help="Iteration of the outer loop")
    parser.add_argument('--samples', type=int, default=100,
                        help="True env samples per iteration (between model updates)")
    parser.add_argument('--posterior_samples', type=int, default=200,
                        help="Number of samples drawn from prior over models to approximate posterior over values")
    parser.add_argument('--gamma', type=float, default=0.99, help="Discount factor")
    parser.add_argument('--lr', type=float, default=1., help="Policy learning rate")
    parser.add_argument('--temp', type=float, default=1., help="Temperature of softmax policy")
    parser.add_argument('--tolerance', type=float, default=1e-3, help="Inner optimization tolerance")
    parser.add_argument('--lambda_', type=float, default=0., help="Regularization coefficient")
    parser.add_argument('--lambda_schedule', type=str, default="const", choices=["const", "linear", "exponential"],
                        help="Regularization coefficient schedule")
    parser.add_argument('--lambda_exponent', type=float, default=1.,
                        help="Exponent for the exponential lambda schedule")
    parser.add_argument('--verbose', type=int, default=10, help="Print logs")
    parser.add_argument('--period', type=int, default=10, help="Frequency of distribution plot")
    parser.add_argument('--show', action="store_true", help="Display plot at the end")
    parser.add_argument('--suffix', type=str, default='', help="Last folder of logs")
    parser.add_argument('--root_dir', type=str, default='vd_pg_tests_2', help="First folder of logs")

    "upper_cvar"  # ["max", "upper_cvar", "upper_delta", "pg"]
    "cvar"  # ["cvar", "lower_bound"]
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    seed = args.seed
    if seed == 0:
        np.random.seed()
        seed = np.random.randint(0, 1000000)
    np.random.seed(seed)
    random.seed(seed)
    args.seed = seed

    gamma = args.gamma
    environment = args.environment
    use_softmax = args.use_softmax
    use_logistic = args.use_logistic
    max_iters = args.max_iters
    period = args.period
    learning_rate = args.lr
    tolerance = args.tolerance
    lambda_lr = 1.  # not used, just for compatibility
    init_lambda = 0  # not used, just for compatibility
    clip_bonus = False  # not used, just for compatibility
    use_adam = False  # not used, just for compatibility
    use_incorrect_priors = False  # not used, just for compatibility
    temp = args.temp
    iterations = args.iterations
    environment_samples_per_iteration = args.samples
    reset_policy = args.reset_policy
    resample = args.resample
    num_posterior_samples = args.posterior_samples
    lambda_ = args.lambda_
    lambda_exponent = args.lambda_exponent
    objective_type = args.objective
    regularization = args.regularization
    lambda_schedule = args.lambda_schedule
    alpha = args.alpha
    delta = args.delta
    verbose = args.verbose
    suffix = args.suffix
    root_dir = args.root_dir
    results = []
    distributions = []
    initial_distributions = []
    save_dir = args.base_dir + "outputs/" + root_dir + "/" + environment + "/" + objective_type + "/" + regularization \
               + "/" + lambda_schedule + "/lambda_" + str(lambda_)[:4]\
               + "/alpha_" + str(alpha)[:4] + "/reset_policy_" + str(reset_policy) + "/post_samples_" \
               + str(num_posterior_samples) + "_delta_" + str(delta) + "_resample_" + str(resample) \
               + "_lr_" + str(learning_rate)[:4]
    save_dir += suffix
    save_dir += "/s" + str(seed) + "/"

    if environment == "loop":
            env = DoubleLoop(seed=seed, gamma=gamma)
            ylims = (-100, 100)
    elif environment == "chain":
            env = Chain(discount=gamma, seed=seed)
            ylims = (0, 800)
    elif environment == "sixarms":
            env = SixArms(gamma=gamma, seed=seed)
            ylims = (0, 800)
    elif environment == "threearms":
            env = TreeArms(gamma=gamma, seed=seed)
            ylims = (0, 800)
    elif environment == "widenarrow":
            env = WideNarrow(gamma=gamma, seed=seed)
            ylims = (0, 800)
    elif environment == "2_state":
            env = SimpleMDP(gamma=gamma, seed=seed)
            ylims = (0, 800)
    elif environment == "grid":
            env = GridWorld(gamma=gamma, seed=seed)
            ylims = (0, 800)
    else:
            raise ValueError("Env not implemented:" + environment)

    if lambda_schedule == "const":
        lambda_param = Parameter(value=lambda_)
    elif lambda_schedule == "linear":
        lambda_param = LinearParameter(value=lambda_, threshold_value=0., n=iterations)
    elif lambda_schedule == "exponential":
        lambda_param = ExponentialParameter(value=lambda_, exp=lambda_exponent)
    else:
        raise ValueError("Lambda schedule not implemented:" + lambda_schedule)


    nState = env.nState
    nAction = env.nAction

    if use_softmax:
            policy = SoftmaxPolicy(nState, nAction, temp, seed)
    elif use_logistic:
            policy = LogisticPolicy(nState, nAction, temp, seed)
    else:
            policy = Policy(nState, nAction, temp, seed)

    true_R = env.R
    true_P = env.P
    initial_distribution = env.initial_distribution

    _, V = value_iteration(P=true_P, R=true_R, gamma=gamma, max_iter=100000, tol=1e-10, qs=False)
    optimal_performance = initial_distribution @ V

    agent = DirichletModel(
        nState,
        nAction,
        int(seed),
        gamma,
        initial_distribution,
        init_lambda,
        lambda_lr,
        learning_rate,
        use_incorrect_priors,
        use_softmax=use_softmax,
        use_logistic=use_logistic,
        use_adam=use_adam,
        clip_bonus=clip_bonus
    )

    n_rows = 1
    n_cols = 1
    # save results
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(save_dir + '/params.json', 'w', encoding='utf-8') as f:
        json.dump(vars(args), f)
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 15))
    ax.set_xlim(0, iterations)
    ax.set_ylim(ylims[0], ylims[1])
    x = np.array([x for x in range(iterations)]) * environment_samples_per_iteration
    line_perfos, = ax.plot([], [], label=" policy return", c="cyan")
    line_lower_bounds, = ax.plot([], [], label=" lower_bound", c="red")
    line_upper_bounds, = ax.plot([], [], label=" upper_bound", c="green")
    line_posterior_means, = ax.plot([], [], label="posterior_mean", c="orange")
    # line_objectives, = ax.plot([], [], label="opt_objective", c="yellow")
    ax.plot([x[0], x[-1]], [optimal_performance, optimal_performance], label="optimal", c="black", linestyle='--')
    ax.set_xlabel("True Environment steps")
    ax.set_ylabel("Return")
    ax.legend()
    for i in range(iterations):
        collect_samples_and_update_prior(agent, env, environment_samples_per_iteration)
        lower_bound, var_alpha, cvar_alpha, avg_performance, cvar_delta, var_delta, upper_bound, grad_norm, converged,\
            it, U_pi_j, initial_distribution, objective = policy_optimization(agent=agent, policy_performance=policy_performance,
                                     num_posterior_samples=num_posterior_samples, objective_type=objective_type,
                                     regularization=regularization, lambda_=lambda_param(), alpha=alpha, delta=delta,
                                     optimization_iterations=max_iters, optimization_tolerance=tolerance,
                                     policy_lr=learning_rate, reset_policy=reset_policy, resample=resample,
                                     )
        lambda_param.update()
        mdp = (true_R, true_P)
        policy_perfo = agent._policy_performance(mdp, agent.policy.get_params())
        results.append([lower_bound, var_alpha, cvar_alpha, avg_performance, cvar_delta, var_delta, upper_bound,
                        grad_norm, converged, it, objective, policy_perfo])
        distributions.append(U_pi_j)
        initial_distributions.append(initial_distribution)
        if verbose:
            print("Finished Iteration:" + str(i + 1))
        results_array = np.array(results)
        np.save(save_dir + "/results.npy", results_array)
        np.save(save_dir + "/distributions.npy", np.array(distributions))
        np.save(save_dir + "/initial_distributions.npy", np.array(initial_distributions))
        lower_bounds = results_array[:, 0]
        cvar_alphas = results_array[:, 2]
        avg_post_performances = results_array[:, 3]
        cvar_deltas = results_array[:, 4]
        upper_bounds = results_array[:, 6]
        policy_perfos = results_array[:, -1]
        objectives = results_array[:, -2]
        x = np.array([x for x in range(len(lower_bounds))]) * environment_samples_per_iteration
        line_perfos.set_xdata(x)
        line_perfos.set_ydata(policy_perfos)

        line_lower_bounds.set_xdata(x)
        line_lower_bounds.set_ydata(lower_bounds)

        line_upper_bounds.set_xdata(x)
        line_upper_bounds.set_ydata(upper_bounds)

        line_posterior_means.set_xdata(x)
        line_posterior_means.set_ydata(avg_post_performances)

        # line_upper_bounds.set_xdata(x)
        # line_upper_bounds.set_ydata(objectives)
        # ax.plot(x, policy_perfos, label=" policy return", c="cyan")
        # ax.plot(x, lower_bounds, label=" lower_bound", c="red")
        # ax.plot(x, upper_bounds, label=" upper_bound", c="green")
        # ax.plot(x, avg_post_performances, label="posterior_mean", c="orange")
        # ax.plot(x, objectives, label="opt_objective", c="yellow")
        # ax.plot([x[0], x[-1]], [optimal_performance, optimal_performance], label="optimal", c="black", linestyle='--')
        ax.set_xlim(0, np.max(x))
        ax.set_ylim(np.min(lower_bounds), np.max(upper_bounds))

        fig.canvas.draw()
        fig.canvas.flush_events()
        fig.savefig(save_dir + '/pg_curve.pdf')
        fig.savefig(save_dir + '/pg_curve.png')

    if args.show:
        plt.show()
    n_rows = len(distributions) // period + 1
    n_cols = 2
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(30, 15))
    x_min = np.inf
    x_max = -np.inf
    for i in range(len(distributions)):
        if i % period != 0:
            continue
        values = np.array(distributions[i])
        initial_values = np.array(initial_distributions[i])

        avg_performance, cvar_alpha, cvar_delta, var_alpha, var_delta, upper_bound, lower_bound, objective, \
            regularization_term, _, _ = \
            compute_statistics(values=values, objective_type=objective_type, regularization=regularization, grads=None)

        ax = axs[i // period][1]
        _x_min = np.min(values)
        _x_max = np.max(values)
        if _x_min < x_min:
            x_min = _x_min
        if _x_max > x_max:
            x_max = _x_max
        ax.set_title("Iteration " + str(i) + "final distribution")
        ax.axvline(regularization_term, c="orange")
        ax.axvline(objective, c="green")
        ax.axvline(objective + lambda_ * regularization_term, c="blue")
        sns.distplot(values, ax=ax, kde=True, label="iteration_" + str(i), color='c')
        sns.distplot(initial_values, ax=ax, kde=True, label="iteration_" + str(i), color='purple',
                     hist_kws=dict(alpha=0.3))

        _x_min = np.min(initial_values)
        _x_max = np.max(initial_values)
        avg_performance, cvar_alpha, cvar_delta, var_alpha, var_delta, upper_bound, lower_bound, objective, \
            regularization_term, grad, constraint_grad = compute_statistics(values=initial_values,
                                                                            objective_type=objective_type,
                                                                            regularization=regularization)
        if _x_min < x_min:
            x_min = _x_min
        if _x_max > x_max:
            x_max = _x_max
        ax = axs[i // period][0]
        ax.set_title("Iteration " + str(i) + "initial distribution")
        ax.axvline(regularization_term, c="orange")
        ax.axvline(objective, c="green")
        ax.axvline(objective + lambda_ * regularization_term, c="blue")
        sns.distplot(initial_values, ax=ax, kde=True, label="iteration_" + str(i), color='c')

    spread = x_max - x_min
    x_min -= 0.15 * spread
    x_max += 0.15 * spread
    for k in range(n_rows):
        axs[k][0].set_xlim(x_min, x_max)
        axs[k][1].set_xlim(x_min, x_max)

    custom_lines = [Line2D([0], [0], color="green", lw=4),
                    Line2D([0], [0], color="orange", lw=4),
                    Line2D([0], [0], color="blue", lw=4)]
    fig.legend(custom_lines, ["obj_func", "regularization", "regularized_obj"], prop={'size': 25}, loc='upper center')
    fig.savefig(save_dir + "/distributions.pdf")
    fig.savefig(save_dir + "/distributions.png")
