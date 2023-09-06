import numpy as np
import jax.numpy as jnp
from jax.scipy.linalg import lu_factor, lu_solve
import jax
from src.utils import get_log_policy, get_policy, get_policy_logistic
from typing import Tuple
from src.adam import Adam
import matplotlib.pyplot as plt

nState = 2
nAction = 2
initial_distribution = np.array([1., 0.])
discount = 0.99


def policy_evaluation(mdp: Tuple[np.ndarray], policy: jnp.ndarray, nState, discount):
    # Vals[state, timestep]
    r_matrix, p_matrix = mdp
    ppi = jnp.einsum('sat,sa->st', p_matrix, policy)
    rpi = jnp.einsum('sa,sa->s', r_matrix, policy)
    v_pi = jnp.linalg.solve(np.eye(nState) -
                            discount*ppi, rpi)
    return v_pi


def policy_performance(r, p, policy_params, initial_distribution, nState, nAction, discount):
    policy = get_policy_logistic(policy_params, nState, nAction)
    vf = policy_evaluation((r, p), policy, nState, discount)
    return initial_distribution @ vf


def posterior_sampling(p_params, num_samples_plan, risk_threshold, R_j=None, P_j=None):
    total_samples = 0
    grad_perf = jax.value_and_grad(policy_performance, 2)
    vmap_grad = jax.vmap(grad_perf, in_axes=(0, 0, None, None, None, None, None))
    U_pi_j, U_pi_j_grad = jax.lax.stop_gradient(vmap_grad(R_j, P_j, p_params, initial_distribution,
                                                          nState, nAction, discount))
    U_pi = np.asarray(U_pi_j)
    U_pi_grads = np.asarray(U_pi_j_grad)
    total_samples += num_samples_plan
    L_pi = U_pi.shape[0]
    sorted_U_pi = np.sort(U_pi)
    U_pi = np.asarray(U_pi)
    U_pi_grads = np.asarray(U_pi_grads)
    floor_index = int(np.floor(risk_threshold * L_pi))
    var_alpha = sorted_U_pi[floor_index]
    cvar_alpha = np.mean(sorted_U_pi[:floor_index])
    return U_pi, U_pi_grads, var_alpha, cvar_alpha, total_samples

constraint = -100
dir = "outputs/"

current_theta = 0.
returns = np.load(dir + "returns.npy")
grads = np.load(dir + "grads.npy")
R_j = np.load(dir + "rewards.npy")
P_j = np.load(dir + "models.npy")

num_samples_plan = P_j.shape[0]
risk_threshold = 0.1
interval = 10
lambda_param = 1.
policy_lr = 0.01
n_discretized = 500
adam_optimizer = Adam(1)
thetas = np.linspace(start=current_theta - interval, stop=current_theta + interval, num=n_discretized)

objectives = []
opt_grads = []
grads = []
cvars = []
for theta in thetas:
    U_pi, U_pi_grads, var_alpha, cvar_alpha, samples_taken = posterior_sampling(theta, num_samples_plan,
                                                                                         risk_threshold, R_j=R_j,
                                                                                         P_j=P_j)
    one_indices = np.nonzero(U_pi < var_alpha)
    cvar_grad_terms = U_pi_grads[one_indices]
    avg_term = 1. / (risk_threshold * num_samples_plan)
    constraint_grad = avg_term * np.sum(cvar_grad_terms, axis=0)
    av_vpi = np.mean(U_pi)
    objective = 0
    upper_risk_threshold = risk_threshold
    optimistic_grad = U_pi_grads[np.argmax(U_pi)]
    objective = np.max(U_pi)

    damp = 0
    p_grad = optimistic_grad + (lambda_param - damp) * constraint_grad
    # step = adam_optimizer.update(p_grad, policy_lr)
    # p_params = p_params + step
    # grad_norm = np.linalg.norm(p_grad)
    # self.policy.update_params(p_params)

    objectives.append(objective)
    opt_grads.append(optimistic_grad)
    grads.append(p_grad)
    cvars.append(cvar_alpha)

visited_thetas = [current_theta]

theta = current_theta

# fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(15, 10))
# axs[0].plot(thetas, objectives)
# axs[0].set_title("Objective")
# axs[1].plot(thetas, opt_grads)
# axs[1].set_title("Objective grad")
# axs[2].plot(thetas, grads)
# axs[2].set_title("Constrained grad")
# axs[3].plot(thetas, cvars)
# axs[3].plot([current_theta - interval, current_theta + interval], [constraint, constraint])
# axs[3].set_title("cvars")
#
# plt.show()

out_path = "func_plots/"

for i in range(100):
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(15, 10))
    U_pi, U_pi_grads, var_alpha, cvar_alpha, samples_taken = posterior_sampling(theta, num_samples_plan,
                                                                                risk_threshold, R_j=R_j,
                                                                                P_j=P_j)
    one_indices = np.nonzero(U_pi < var_alpha)
    cvar_grad_terms = U_pi_grads[one_indices]
    avg_term = 1. / (risk_threshold * num_samples_plan)
    constraint_grad = avg_term * np.sum(cvar_grad_terms, axis=0)
    av_vpi = np.mean(U_pi)
    objective = 0
    upper_risk_threshold = risk_threshold
    optimistic_grad = U_pi_grads[np.argmax(U_pi)]
    objective = np.max(U_pi)

    damp = 0
    p_grad = optimistic_grad + (lambda_param - damp) * constraint_grad
    # step = adam_optimizer.update(p_grad, policy_lr)
    step = policy_lr * p_grad
    old_theta = theta
    theta = theta + step
    print("Theta " + str(i+1) + ":" + str(theta))
    # grad_norm = np.linalg.norm(p_grad)
    visited_thetas.append(theta)
    axs[0].plot(thetas, objectives, c="blue")
    axs[0].set_title("Objective")
    axs[1].plot(thetas, opt_grads, c="blue")
    axs[1].set_title("Objective grad")
    axs[2].plot(thetas, grads, c="blue")
    axs[2].set_title("Constrained grad")
    axs[3].plot(thetas, cvars, c="blue")
    axs[3].plot([current_theta - interval, current_theta + interval], [constraint, constraint], c="orange")
    axs[3].set_title("cvars")
    to_plot = [objective, optimistic_grad, p_grad, cvar_alpha]
    for j, ax in enumerate(axs):
        ax.scatter([old_theta], [to_plot[j]], c="blue")

    # fig.savefig(out_path + str(i) + "_functions.png")
    plt.savefig(out_path + str(i) + "_functions.png")  # write image to file
    plt.close(fig)
    # input()


#

