import sys
sys.path.append("..")
import matplotlib.pyplot as plt
import glob
import numpy as np
import pickle
import os

colors = ['c', 'k', 'orange', 'purple', 'r', 'b', 'g', 'y', 'brown', 'magenta', '#BC8D0B', "#006400"]
markers = ['o', 's', 'v', 'D', 'x', '*', '|', '+', '^', '2', '1', '3', '4']
LEGEND_FONT_SIZE = 28
AXIS_FONT_SIZE = 28
TICKS_FONT_SIZE = 26
MARKER_SIZE = 10
LINE_WIDTH = 3.0
TITLE_SIZE= 28

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

    dirs = ["../outputs/vd_pg_presentation/chain/pg/cvar/lambda_0.0/alpha_0.1/reset_policy_True/post_samples_20_delta_0.9_resample_True_lr_0.1",
            "../outputs/vd_pg_presentation/chain/pg/cvar/lambda_0.0/alpha_0.1/reset_policy_True/post_samples_50_delta_0.9_resample_True_lr_0.1",
            "../outputs/vd_pg_presentation/chain/pg/cvar/lambda_0.0/alpha_0.1/reset_policy_True/post_samples_100_delta_0.9_resample_True_lr_0.1",
            # "../outputs/vd_pg_presentation/chain/pg/cvar/lambda_0.0/alpha_0.1/reset_policy_True/post_samples_500_delta_0.9_resample_True_lr_0.1",
            # "../outputs/vd_pg_presentation/chain/pg/cvar/lambda_0.0/alpha_0.1/reset_policy_True/post_samples_1000_delta_0.9_resample_True_lr_0.1",
            ]

    labels = ["chain_samples_20", "chain_samples_50", "chain_samples_100",
              # "chain_samples_500", "chain_samples_1000"
              ]

    dirs = ["../outputs/vd_pg_presentation/chain/pg/cvar/lambda_0.0/alpha_0.1/reset_policy_True/post_samples_1000_delta_0.9_resample_True_lr_0.01",
            "../outputs/vd_pg_presentation/chain/max/cvar/lambda_0.0/alpha_0.01/reset_policy_True/post_samples_1000_delta_0.9_resample_True_lr_0.1",
            "../outputs/vd_pg_presentation/chain/max/cvar/lambda_0.1/alpha_0.01/reset_policy_True/post_samples_1000_delta_0.9_resample_True_lr_0.1",
            "../outputs/vd_pg_presentation/chain/max/cvar/lambda_0.01/alpha_0.01/reset_policy_True/post_samples_1000_delta_0.9_resample_True_lr_0.1",
            "../outputs/vd_pg_presentation/chain/max/cvar/lambda_0.01/alpha_0.1/reset_policy_True/post_samples_1000_delta_0.9_resample_True_lr_0.1",
            "../outputs/vd_pg_presentation/chain/max/cvar/lambda_0.01/alpha_0.25/reset_policy_True/post_samples_1000_delta_0.9_resample_True_lr_0.1",
            ]
    labels = ["chain_pg", "chain_max", "chain_max_cvar", "chain_max_smaller_cvar", "chain_max_smaller_cvar_higher_alpha",
              "chain_max_smaller_cvar_even_higher_alpha"]
    dirs = [
            "../outputs/vd_pg_presentation/chain/max/lower_bound/lambda_0.01/alpha_0.1/reset_policy_True/post_samples_1000_delta_0.9_resample_True_lr_0.1",
            "../outputs/vd_pg_presentation/chain/max/lower_bound/lambda_0.01/alpha_0.01/reset_policy_True/post_samples_1000_delta_0.9_resample_True_lr_0.1",
            "../outputs/vd_pg_presentation/chain/max/lower_bound/lambda_0.01/alpha_0.25/reset_policy_True/post_samples_1000_delta_0.9_resample_True_lr_0.1",
            "../outputs/vd_pg_presentation/chain/max/lower_bound/lambda_0.1/alpha_0.1/reset_policy_True/post_samples_1000_delta_0.9_resample_True_lr_0.1",
            "../outputs/vd_pg_presentation/chain/max/lower_bound/lambda_0.1/alpha_0.01/reset_policy_True/post_samples_1000_delta_0.9_resample_True_lr_0.1",
            "../outputs/vd_pg_presentation/chain/max/lower_bound/lambda_0.1/alpha_0.25/reset_policy_True/post_samples_1000_delta_0.9_resample_True_lr_0.1",

    ]
    labels = ["chain_max_lb_low_reg_small_a_0.1", "chain_max_lb_low_reg_small_a_0.01", "chain_max_lb_low_reg_small_a_0.25",
              "chain_max_lb_lg_reg_small_a_0.1", "chain_max_lb_lg_reg_small_a_0.01", "chain_max_lb_lg_reg_small_a_0.25"]
    out_dir = "chain_sample_ablation/"
    dirs = [
        "../outputs/vd_pg_presentation/chain/pg/cvar/lambda_0.0/alpha_0.1/reset_policy_True/post_samples_1000_delta_0.9_resample_True_lr_0.01",
        "../outputs/vd_pg_presentation/chain/max/cvar/lambda_0.0/alpha_0.01/reset_policy_True/post_samples_1000_delta_0.9_resample_True_lr_0.1",
        "../outputs/vd_pg_presentation/chain/max/cvar/lambda_0.1/alpha_0.01/reset_policy_True/post_samples_1000_delta_0.9_resample_True_lr_0.1",
        "../outputs/vd_pg_presentation/chain/max/cvar/lambda_0.01/alpha_0.01/reset_policy_True/post_samples_1000_delta_0.9_resample_True_lr_0.1",
        "../outputs/vd_pg_presentation/chain/max/cvar/lambda_0.01/alpha_0.1/reset_policy_True/post_samples_1000_delta_0.9_resample_True_lr_0.1",
        "../outputs/vd_pg_presentation/chain/max/cvar/lambda_0.01/alpha_0.25/reset_policy_True/post_samples_1000_delta_0.9_resample_True_lr_0.1",
        ]
    labels = ["chain_pg",
              "chain_max",
              # "chain_max_cvar", "chain_max_smaller_cvar",
              # "chain_max_smaller_cvar_higher_alpha",
              # "chain_max_smaller_cvar_even_higher_alpha"
              ]
    out_dir = "pgs/"
    suffix = "chain"
    dirs = ["../outputs/vd_pg_presentation/loop/pg/cvar/lambda_0.0/alpha_0.1/reset_policy_True/post_samples_1000_delta_0.9_resample_True_lr_0.1",
            "../outputs/vd_pg_presentation/loop/max/cvar/lambda_0.0/alpha_0.1/reset_policy_True/post_samples_1000_delta_0.9_resample_True_lr_0.1",
            ]
    labels = ["loop_pg",
              "loop_max"]
    suffix = "loop"
    dirs = [
        "../outputs/vd_pg_presentation/sixarms/pg/cvar/lambda_0.0/alpha_0.1/reset_policy_True/post_samples_1000_delta_0.9_resample_True_lr_0.1",
        "../outputs/vd_pg_presentation/sixarms/max/cvar/lambda_0.0/alpha_0.1/reset_policy_True/post_samples_1000_delta_0.9_resample_True_lr_0.1",
        ]
    labels = ["sixarms_pg",
              "sixarms_max"]
    suffix = "sixarms"
    dirs = [
        "../outputs/vd_pg_presentation/widenarrow/pg/cvar/lambda_0.0/alpha_0.1/reset_policy_True/post_samples_1000_delta_0.9_resample_True_lr_0.1",
        "../outputs/vd_pg_presentation/widenarrow/max/cvar/lambda_0.0/alpha_0.1/reset_policy_True/post_samples_1000_delta_0.9_resample_True_lr_0.1",
    ]
    labels = ["widenarrow_pg",
              "widenarrow_max"]
    suffix = "widenarrow"
    dirs = [
        "../outputs/vd_pg_presentation/2_state/pg/cvar/lambda_0.0/alpha_0.1/reset_policy_True/post_samples_1000_delta_0.9_resample_True_lr_0.1",
        "../outputs/vd_pg_presentation/2_state/max/cvar/lambda_0.0/alpha_0.1/reset_policy_True/post_samples_1000_delta_0.9_resample_True_lr_0.1",
    ]
    labels = ["2_state_pg",
              "2_state_max"]
    suffix = "2_state"

    dirs = [
        "../outputs/vd_pg_presentation/chain/pg/cvar/lambda_0.0/alpha_0.1/reset_policy_True/post_samples_1000_delta_0.9_resample_True_lr_0.01",
        "../outputs/vd_pg_presentation/chain/max/cvar/lambda_0.0/alpha_0.01/reset_policy_True/post_samples_1000_delta_0.9_resample_True_lr_0.1",
        # "../outputs/vd_pg_presentation/chain/max/cvar/lambda_0.1/alpha_0.01/reset_policy_True/post_samples_1000_delta_0.9_resample_True_lr_0.1",
        # "../outputs/vd_pg_presentation/chain/max/cvar/lambda_0.01/alpha_0.01/reset_policy_True/post_samples_1000_delta_0.9_resample_True_lr_0.1",
        "../outputs/vd_pg_presentation/chain/max/cvar/lambda_0.01/alpha_0.1/reset_policy_True/post_samples_1000_delta_0.9_resample_True_lr_0.1",
        "../outputs/vd_pg_presentation/chain/max/lower_bound/lambda_0.01/alpha_0.1/reset_policy_True/post_samples_1000_delta_0.9_resample_True_lr_0.1",
    ]
    labels = ["chain_pg",
              "chain_max",
              # "chain_max_cvar", "chain_max_smaller_cvar",
              "chain_max_cvar",
              "chain_max_var"
              ]
    out_dir = "cvars/"
    suffix = "chain"

    dirs = [
        "../outputs/vd_pg_presentation/chain/max/lower_bound/lambda_0.01/alpha_0.01/reset_policy_True/post_samples_1000_delta_0.9_resample_True_lr_0.1",
        "../outputs/vd_pg_presentation/chain/max/lower_bound/lambda_0.01/alpha_0.1/reset_policy_True/post_samples_1000_delta_0.9_resample_True_lr_0.1",
        "../outputs/vd_pg_presentation/chain/max/lower_bound/lambda_0.01/alpha_0.25/reset_policy_True/post_samples_1000_delta_0.9_resample_True_lr_0.1",
        # "../outputs/vd_pg_presentation/chain/max/lower_bound/lambda_0.1/alpha_0.01/reset_policy_True/post_samples_1000_delta_0.9_resample_True_lr_0.1",
        # "../outputs/vd_pg_presentation/chain/max/lower_bound/lambda_0.1/alpha_0.1/reset_policy_True/post_samples_1000_delta_0.9_resample_True_lr_0.1",
        # "../outputs/vd_pg_presentation/chain/max/lower_bound/lambda_0.1/alpha_0.25/reset_policy_True/post_samples_1000_delta_0.9_resample_True_lr_0.1",

    ]
    labels = [
              "low_l_low_a", "low_l_mid_a", "low_l_high_a",
              # "high_l_low_a", "high_l_mid_a", "high_l_high_a"
              ]
    out_dir = "ablation_lambda/"
    suffix = "chain_low_l"

    dirs = [
        "../outputs/vd_pg_presentation/loop/max/lower_bound/lambda_0.01/alpha_0.01/reset_policy_True/post_samples_1000_delta_0.9_resample_True_lr_0.1",
        "../outputs/vd_pg_presentation/loop/max/lower_bound/lambda_0.01/alpha_0.1/reset_policy_True/post_samples_1000_delta_0.9_resample_True_lr_0.1",
        "../outputs/vd_pg_presentation/loop/max/lower_bound/lambda_0.01/alpha_0.25/reset_policy_True/post_samples_1000_delta_0.9_resample_True_lr_0.1",
        # "../outputs/vd_pg_presentation/loop/max/lower_bound/lambda_0.1/alpha_0.01/reset_policy_True/post_samples_1000_delta_0.9_resample_True_lr_0.1",
        # "../outputs/vd_pg_presentation/loop/max/lower_bound/lambda_0.1/alpha_0.1/reset_policy_True/post_samples_1000_delta_0.9_resample_True_lr_0.1",
        # "../outputs/vd_pg_presentation/loop/max/lower_bound/lambda_0.1/alpha_0.25/reset_policy_True/post_samples_1000_delta_0.9_resample_True_lr_0.1",

    ]
    labels = [
        "low_l_low_a", "low_l_mid_a", "low_l_high_a",
        # "high_l_low_a", "high_l_mid_a", "high_l_high_a"
    ]
    out_dir = "ablation_alpha/"
    suffix = "loop_low_l"
    env = dirs[0].split("/")[3]
    separate = False
    more_seeds = True
    N = 1
    environment_samples_per_iteration = 100
    with open('../optimals.pkl', 'rb') as inp:
        optimals = pickle.load(inp)
    optimal_performance = optimals[env]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
    for d, dir in enumerate(dirs):
        label = labels[d]
        if more_seeds:
            paths = glob.glob(dir + "/*/results.npy")
            n = len(paths)
            if n == 0:
                print("No paths found in:" + dir)
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
            ax.plot(x, performance, label=label, c=colors[d])
        else:
            for k, result in enumerate(results):
                performance = result[:, -1]
                x = np.arange(performance.shape[0]) * environment_samples_per_iteration
                performance = running_mean(performance, N)
                seed = paths[k].split("/")[-2]
                ax.plot(x, performance, label=label + "_" + seed, c=colors[d])
        ax.set_xlabel("True Environment steps", fontsize=AXIS_FONT_SIZE)
        ax.set_ylabel("Return", fontsize=AXIS_FONT_SIZE)
        if n > 1 and not separate:
            ax.fill_between(x, performance - 2 * performance_error,
                            performance + 2 * performance_error,
                            alpha=0.2, color=colors[d])

    ax.plot([x[0], x[-1]], [optimal_performance, optimal_performance], label="optimal", c="black",
            linestyle='--')
    fig.legend( prop={'size': 25}, ncols=2)

    ax.tick_params(labelsize=TICKS_FONT_SIZE)
    ax.ticklabel_format(useOffset=False, style='plain')
    fig.suptitle(env)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fig.savefig(out_dir + '/' + suffix + '_result.pdf')
    fig.savefig(out_dir + '/' + suffix + '_result.png')
    plt.show()