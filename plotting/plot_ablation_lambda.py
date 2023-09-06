import sys
sys.path.append("..")
import os
import matplotlib.pyplot as plt
import glob
import numpy as np

from vd_pg import parse_args


colors = ['c', 'k', 'orange', 'purple', 'r', 'b', 'g', 'y', 'brown', 'magenta', '#BC8D0B', "#006400"]
markers = ['o', 's', 'v', 'D', 'x', '*', '|', '+', '^', '2', '1', '3', '4']


def running_mean(x, N):
	divider = np.convolve(np.ones_like(x), np.ones((N,)), mode='same')
	return np.convolve(x, np.ones((N,)), mode='same') / divider


if __name__ == "__main__":
    args = parse_args()
    N = 5
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
    objective_type = args.objective
    regularization = args.regularization
    alpha = args.alpha
    delta = args.delta
    verbose = args.verbose
    results = []

    lambdas = [0., 0.1, 0.5, 1., 2., 5.]
    resamples = [True, False]
    resets = [True, False]
    envs = ["loop", "chain"]
    objectives = ["max", "upper_cvar", "upper_delta"]
    regularizations = ["cvar", "lower_bound"]

    for environment in envs:
        for objective in objectives:
            for regularization in regularizations:
                out_dir = "../outputs/vd_pg/" + environment + "/" + objective_type + "/" + regularization + \
                          "/lambda_ablation/"
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))

                for i, lambda_ in enumerate(lambdas):
                    save_dir = "../outputs/vd_pg/" + environment + "/" + objective_type + "/" + regularization + "/lambda_" + \
                               str(lambda_)[:4] + "/alpha_" + str(alpha)[:4] + "/reset_policy_" + str(reset_policy) + "/post_samples_" \
                               + str(num_posterior_samples) + "_delta_" + str(delta) + "_resample_" + str(resample) + \
                               "" #"_lr_" + str(learning_rate)[:4]
                    paths = glob.glob(save_dir + "/*/results.npy")
                    n = len(paths)
                    if n == 0:
                        print("No paths found for lambda " + str(lambda_))
                        print(save_dir + "/*/results.npy")
                        continue
                    results = []
                    for path in paths:
                        result = np.load(path)
                        results.append(result)
                    results = np.stack(results, axis=0)
                    results_mean = np.mean(results, axis=0)
                    results_error = np.std(results, axis=0) / np.sqrt(n)
                    print(results_mean.shape)
                    performance = results_mean[:, -1]
                    performance_error = results_error[:, -1]
                    x = np.arange(performance.shape[0])
                    performance = running_mean(performance, N)
                    performance_error = running_mean(performance_error, N)
                    ax.plot(x, performance, label="lambda-" + str(lambda_)[:4], c=colors[i])
                    if n > 1:
                        ax.fill_between(x, performance - 2 * performance_error,
                                        performance + 2 * performance_error,
                                        alpha=0.2, color=colors[i])
                fig.legend()
                fig.suptitle(environment)
                fig.savefig(out_dir + '/lambda_curves.pdf')
                fig.savefig(out_dir + '/lambda_curves.png')