from envs.doubleloop import DoubleLoop
from envs.chain import Chain
from envs.six_arms import SixArms
from envs.three_arms import TreeArms
from envs.wide_narrow import WideNarrow
from envs.basic_envs import SimpleMDP
import numpy as np
from src.utils import value_iteration
import pickle

seed = 0
gamma = 0.99
envs = [DoubleLoop(seed=seed, gamma=gamma),
        Chain(discount=gamma, seed=seed),
        SixArms(gamma=gamma, seed=seed),
        TreeArms(gamma=gamma, seed=seed),
        WideNarrow(gamma=gamma, seed=seed),
        SimpleMDP(gamma=gamma, seed=seed)]
env_labels = ["loop", "chain", "sixarms", "threearms", "widenarrow", "2_state"]
optimals = {}
for i, env in enumerate(envs):
    true_R = env.R
    true_P = env.P
    initial_distribution = env.initial_distribution

    _, V = value_iteration(P=true_P, R=true_R, gamma=gamma, max_iter=100000, tol=1e-10, qs=False)
    optimal_performance = initial_distribution @ V
    optimals[env_labels[i]] = optimal_performance


with open('optimals.pkl', 'wb') as outp:
    pickle.dump(optimals, outp, pickle.HIGHEST_PROTOCOL)


