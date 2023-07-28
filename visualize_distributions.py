import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re
import glob
dir_path = "outputs/twoState_max-opt-cvar_constrained_incorrectpriorsFalse/1690477398.6798425"
dir_path = "outputs/twoState_max-opt-cvar_constrained_incorrectpriorsFalse/1690481413.003639"
dir_path = "outputs/twoState_max-opt-cvar_constrained_incorrectpriorsFalse/1690482231.4659538"
dir_path = "outputs/twoState_max-opt-cvar_constrained_incorrectpriorsFalse/1690491626.880182"
dir_path = "outputs/CliffWalking-v0_max-opt-cvar_constrained_incorrectpriorsFalse/1690557862.417146"
dir_path = "outputs/twoState_max-opt-cvar_constrained_incorrectpriorsFalse/1690568125.6617167"
dir_path = "outputs/twoState_max-opt-cvar_constrained_incorrectpriorsFalse/1690569595.4249392"
dir_path = "outputs/CliffWalking-v0_max-opt-cvar_constrained_incorrectpriorsFalse/1690569639.8797767"
dir_path = "outputs/twoState_max-opt-cvar_constrained_incorrectpriorsFalse/1690575854.2166948"

paths = glob.glob(dir_path + "/*.npy")
paths.remove(dir_path + "/cvars.npy")
paths.remove(dir_path + "/contraints.npy")
paths.remove(dir_path + "/objectives.npy")
period = 1
risk_threshold = 0.1
cvars = np.load(dir_path + "/cvars.npy")
contraints = np.load(dir_path + "/contraints.npy")
objectives = np.load(dir_path + "/objectives.npy")
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
x = np.array([x for x in range(len(cvars))]) * period
ax.plot(x, cvars, label="cvar")
ax.plot(x, contraints, label="constrain")
ax.plot(x, objectives, label="objective")
ax.set_xlabel("iteration")
lgd = ax.legend()
fig.savefig(dir_path + '/curve_training.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
if len(paths) > 0:
    n_rows = len(paths) // period + 1
    n_cols = 1
    paths.sort(key=lambda f: int(re.sub('\D', '', f)))
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(30, 15))
    x_min = np.inf
    x_max = -np.inf
    for i, path in enumerate(paths):
        values = np.load(path)
        m = len(values)
        sorted_values = np.sort(values)
        floor_index = int(np.floor(risk_threshold * m))
        var_alpha = sorted_values[floor_index]
        cvar_alpha = np.mean(sorted_values[:floor_index])
        objective = np.max(values)
        if i % period != 0:
            continue
        ax = axs[i // period]
        _x_min = np.min(values)
        _x_max = np.max(values)
        if _x_min < x_min:
            x_min = _x_min
        if _x_max > x_max:
            x_max = _x_max
        ax.set_title("Iteration " + str(i))
        constrain = contraints[i]
        ax.axvline(constrain, c="red")
        ax.axvline(cvar_alpha, c="green")
        ax.axvline(objective, c="orange")
        sns.distplot(values, ax=ax, kde=True, label="iteration_" + str(i), color='c')
    spread = x_max - x_min
    x_min -= 0.15 * spread
    x_max += 0.15 * spread
    for ax in axs:
        ax.set_xlim(x_min, x_max)
    fig.savefig(dir_path + "/distributions.pdf")

