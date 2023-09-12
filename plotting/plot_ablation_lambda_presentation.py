import sys
sys.path.append("..")
import os
import matplotlib.pyplot as plt
import glob
import numpy as np
import pickle

from vd_pg import parse_args


colors = ['c', 'k', 'orange', 'purple', 'r', 'b', 'g', 'y', 'brown', 'magenta', '#BC8D0B', "#006400"]
markers = ['o', 's', 'v', 'D', 'x', '*', '|', '+', '^', '2', '1', '3', '4']


def running_mean(x, N):
    divider = np.convolve(np.ones_like(x), np.ones((N,)), mode='same')
    return np.convolve(x, np.ones((N,)), mode='same') / divider


if __name__ == "__main__":
    args = parse_args()
    N = 1
    base_dir = "vd_pg_presentation/"
    base_out_dir = "plots_presentation/"
    gamma = args.gamma
    environment = args.environment
    use_softmax = args.use_softmax
    use_logistic = args.use_logistic
    max_iters = 50
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
    reset_policy = True
    resample = True
    num_posterior_samples = 1000
    lambda_ = args.lambda_
    objective_type = args.objective
    regularization = args.regularization
    alpha = 0.1
    delta = args.delta
    verbose = args.verbose
    suffix = args.suffix
    results = []

    lambdas = [ 0., 0.01, 0.1]  #  0., 0.1, 0.5, 2., , 0.5, 1., 2., 5.
    resamples = [True]
    resets = [True]
    envs = ["loop", "chain", "sixarms", "2_state", "widenarrow"]  # "loop", "chain",
    objectives = ["pg", "max"] # "upper_cvar", "upper_delta"
    regularizations = ["cvar", "lower_bound"]
    env_to_lr = {
        "loop": 0.1,
        "chain": 0.1,
        "chain_turing": 0.1,
        "2_state": 0.1,
        "sixarms": 0.1,
        "widenarrow": 0.1,
    }
    with open('../optimals.pkl', 'rb') as inp:
        env_to_optimal = pickle.load(inp)
    for environment in envs:
        optimal_performance = env_to_optimal[environment]
        for objective_type in objectives:
            for regularization in regularizations:
                out_dir = base_out_dir + "/" + environment + "/" + objective_type + "/" + regularization + "/"
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))

                found_lambda = False
                for i, lambda_ in enumerate(lambdas):
                    save_dir = "../outputs/" + base_dir + "/" + environment + "/" + objective_type + "/" + \
                               regularization + "/lambda_" + str(lambda_)[:4] + "/alpha_" + str(alpha)[:4] + \
                               "/reset_policy_" + str(reset_policy) + "/post_samples_" \
                               + str(num_posterior_samples) + "_delta_" + str(delta) + "_resample_" + str(resample) + \
                               "_lr_" + str(env_to_lr[environment])[:4]
                    paths = glob.glob(save_dir + "/*/results.npy")
                    n = len(paths)
                    if n == 0:
                        print("No paths found for lambda " + str(lambda_))
                        print(save_dir + "/*/results.npy")
                        continue
                    results = []
                    min_length = np.inf
                    for path in paths:
                        found_lambda = True
                        result = np.load(path)
                        results.append(result)
                        if result.shape[0] < min_length:
                            min_length = result.shape[0]
                    for k, result in enumerate(results):
                            results[k] = result[:min_length]
                    results = np.stack(results, axis=0)
                    results_mean = np.mean(results, axis=0)
                    results_error = np.std(results, axis=0) / np.sqrt(n)
                    print(results_mean.shape)
                    performance = results_mean[:, -1]
                    performance_error = results_error[:, -1]
                    x = np.arange(performance.shape[0]) * environment_samples_per_iteration
                    performance = running_mean(performance, N)
                    performance_error = running_mean(performance_error, N)
                    ax.plot(x, performance, label=objective_type + "-lambda-" + str(lambda_)[:4], c=colors[i])
                    ax.set_xlabel("True Environment steps")
                    ax.set_ylabel("Return")
                    if n > 1:
                        ax.fill_between(x, performance - 2 * performance_error,
                                        performance + 2 * performance_error,
                                        alpha=0.2, color=colors[i])

                if found_lambda:

                    ax.plot([x[0], x[-1]], [optimal_performance, optimal_performance], label="optimal", c="black",
                            linestyle='--')
                    fig.legend(ncols=(len(lambdas) + 1) // 2, prop={'size': 25})
                    fig.suptitle(environment)
                    fig.savefig(out_dir + '/lambda_curves.pdf')
                    fig.savefig(out_dir + '/lambda_curves.png')