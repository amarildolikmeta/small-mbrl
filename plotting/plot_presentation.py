import sys
sys.path.append("..")
import matplotlib.pyplot as plt
import glob
import numpy as np
import pickle


colors = ['c', 'k', 'orange', 'purple', 'r', 'b', 'g', 'y', 'brown', 'magenta', '#BC8D0B', "#006400"]
markers = ['o', 's', 'v', 'D', 'x', '*', '|', '+', '^', '2', '1', '3', '4']


def running_mean(x, N):
    divider = np.convolve(np.ones_like(x), np.ones((N,)), mode='same')
    return np.convolve(x, np.ones((N,)), mode='same') / divider


if __name__ == "__main__":

    dir = "../outputs/vd_pg_presentation/2_state/pg/cvar/lambda_0.0/alpha_0.1/reset_policy_True/post_samples_200_delta_0.9_resample_True_lr_0.1"
    dir = "../outputs/vd_pg_presentation_2/2_state/pg/cvar/lambda_0.0/alpha_0.1/reset_policy_True/post_samples_200_delta_0.9_resample_True_lr_0.1"
    dir = "../outputs/vd_pg_presentation/chain/pg/cvar/lambda_0.0/alpha_0.1/reset_policy_True/post_samples_100_delta_0.9_resample_True_lr_0.1"
    dir = "../outputs/vd_pg_presentation_2/chain/pg/cvar/lambda_0.0/alpha_0.1/reset_policy_True/post_samples_500_delta_0.9_resample_True_lr_0.1"
    dir = "../outputs/vd_pg_presentation/chain/pg/cvar/lambda_0.0/alpha_0.1/reset_policy_True/post_samples_500_delta_0.9_resample_True_lr_0.01"
    dir = "../outputs/vd_pg_presentation_2/2_state/pg/cvar/lambda_0.0/alpha_0.1/reset_policy_True/post_samples_100_delta_0.9_resample_True_lr_0.1"
    env = dir.split("/")[3]
    separate = False
    more_seeds = True
    N = 1
    environment_samples_per_iteration = 100
    with open('../optimals.pkl', 'rb') as inp:
        optimals = pickle.load(inp)
    optimal_performance = optimals[env]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
    if more_seeds:
        paths = glob.glob(dir + "/*/results.npy")
        n = len(paths)
        if n == 0:
            print("No paths found in:" + dir)
            exit()

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
        if not separate:
            results = np.stack(results, axis=0)
    else:
        path = dir + "/results.npy"
        result = np.load()[None, ]
        n = 1
    if not separate:
        results_mean = np.mean(results, axis=0)
        results_error = np.std(results, axis=0) / np.sqrt(n)
        print(results_mean.shape)
        performance = results_mean[:, -1]
        performance_error = results_error[:, -1]
        x = np.arange(performance.shape[0]) * environment_samples_per_iteration
        performance = running_mean(performance, N)
        performance_error = running_mean(performance_error, N)
        ax.plot(x, performance, label="pg")
    else:
        for k, result in enumerate(results):
            performance = result[:, -1]
            x = np.arange(performance.shape[0]) * environment_samples_per_iteration
            performance = running_mean(performance, N)
            seed = paths[k].split("/")[-2]
            ax.plot(x, performance, label=seed)
    ax.set_xlabel("True Environment steps")
    ax.set_ylabel("Return")
    if n > 1 and not separate:
        ax.fill_between(x, performance - 2 * performance_error,
                        performance + 2 * performance_error,
                        alpha=0.2)

    ax.plot([x[0], x[-1]], [optimal_performance, optimal_performance], label="optimal", c="black",
            linestyle='--')
    fig.legend( prop={'size': 25})
    fig.suptitle(env)
    fig.savefig(dir + '/result.pdf')
    fig.savefig(dir + '/result.png')
    plt.show()