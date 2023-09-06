import matplotlib.pyplot as plt
import numpy as np
import glob
from matplotlib.lines import Line2D

dir_path = "outputs/twoState_max-opt-cvar_constrained_incorrectpriorsFalse/1690477398.6798425"
dir_path = "outputs/twoState_max-opt-cvar_constrained_incorrectpriorsFalse/1690481413.003639"
dir_path = "outputs/twoState_max-opt-cvar_constrained_incorrectpriorsFalse/1690482231.4659538"
dir_path = "outputs/twoState_max-opt-cvar_constrained_incorrectpriorsFalse/1690491626.880182"
dir_path = "outputs/CliffWalking-v0_max-opt-cvar_constrained_incorrectpriorsFalse/1690557862.417146"
dir_path = "outputs/twoState_max-opt-cvar_constrained_incorrectpriorsFalse/1690568125.6617167"
dir_path = "outputs/twoState_max-opt-cvar_constrained_incorrectpriorsFalse/1690569595.4249392"
dir_path = "outputs/CliffWalking-v0_max-opt-cvar_constrained_incorrectpriorsFalse/1690569639.8797767"
dir_path = "outputs/twoState_max-opt-cvar_constrained_incorrectpriorsFalse/1690575854.2166948"
dir_path = "outputs/twoState_max-opt-cvar_constrained_incorrectpriorsFalse/1691604846.3912938"
dir_path = "outputs/DistributionalShift-v0_max-opt-cvar_constrained_incorrectpriorsFalse/1691676375.65314"
dir_path = "outputs/twoState_max-opt-cvar_constrained_incorrectpriorsFalse/"
dir_path = "outputs/chain_max-opt-cvar_constrained_incorrectpriorsFalse/"
# dir_path = "outputs/doubleloop_max-opt-cvar_constrained_incorrectpriorsFalse/"

optimal_value = np.load(dir_path + "/optimal_value.npy")
period = 1
risk_threshold = 0.1
lambdas = []
policy = "softmax_policy"
paths = glob.glob(dir_path + policy + "/lambda_*/*/*.npy")
max_range = -np.inf
min_range = np.inf
for path in paths:
    lambda_param = float(path.split("/")[3].split("_")[-1])
    if lambda_param not in lambdas:
        lambdas.append(lambda_param)
lambdas = sorted(lambdas)
n_cols = 2
n_rows = len(lambdas) // 2 + len(lambdas) % 2

fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 15))
for i, lambda_param in enumerate(lambdas):
    paths = glob.glob(dir_path + policy + "/lambda_" + str(lambda_param) + "/*/*.npy")
    p = "/".join(paths[0].split("/")[:-1])
    paths.remove(p + "/cvars.npy")
    paths.remove(p + "/contraints.npy")
    paths.remove(p + "/objectives.npy")
    paths.remove(p + "/policy_returns.npy")

    cvars = np.load(p + "/cvars.npy")
    contraints = np.load(p + "/contraints.npy")
    objectives = np.load(p + "/objectives.npy")
    policy_returns = np.load(p + "/policy_returns.npy")
    ax = axs[i // n_cols][i % n_cols]
    x = np.array([x for x in range(len(cvars))]) * period
    ax.plot(x[1:], cvars[1:], label="cvar", c="b")
    ax.plot(x[1:], contraints[1:], label="constrain", c="r")
    ax.plot(x[1:], objectives[1:], label="objective", c="g")
    ax.plot(x[1:], policy_returns[1:], label="performance", c="cyan")
    ax.plot([x[1], x[-1]], [optimal_value, optimal_value], label="optimal", c="black", linestyle='--')

    current_max = np.max(np.concatenate([cvars[1:], contraints[1:], objectives[1:], policy_returns[1:]]))
    current_min = np.min(np.concatenate([cvars[1:], contraints[1:], objectives[1:], policy_returns[1:]]))
    if current_max > max_range:
        max_range = current_max
    if current_min < min_range:
        min_range = current_min
    ax.set_title("Lambda " + str(lambda_param))
    # lgd = ax.legend()
for i in range(n_cols):
    axs[-1][i].set_xlabel("iteration")
for i in range(n_rows):
    for j in range(n_cols):
        axs[i][j].set_ylim(min_range, max_range)
custom_lines = [Line2D([0], [0], color="blue", lw=4),
                Line2D([0], [0], color="red", lw=4),
                Line2D([0], [0], color="green", lw=4),
                Line2D([0], [0], color="cyan", lw=4),
                Line2D([0], [0], c="black", linestyle='--', lw=4)]
fig.legend(custom_lines, ["cvar", "constraint", "objective", "performance", "optimal"],
           prop={'size': 25}, ncols=len(custom_lines),
           )

fig.savefig(dir_path + '/curve_training_lambdas.pdf')

