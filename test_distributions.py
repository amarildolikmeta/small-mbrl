from envs.mdp import random_mdp, generate_chain
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class MDPDist:
    def __init__(self, n_states, n_actions, gamma=0.999, distribution=None, weights=None):
        self.distribution = distribution
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.weights = weights

    def sample(self):
        if self.distribution is None:
            return random_mdp(n_states=self.n_states, n_actions=self.n_actions, gamma=self.gamma, two_d_r=True)
        elif isinstance(self.distribution, list):
            weights = self.weights
            if weights is None:
                weights = np.ones(len(self.distribution)) / len(self.distribution)
            return self.distribution[np.random.choice(len(self.distribution), p=weights)]
        else:
            raise ValueError("Have not implemented specific MDP distributions")


def run_random_mdps(nS, nA, gamma, n_samples, factor, episodes=100, run_alternatives=False, run_operator=False):
    effective_horizon = int(np.ceil(1 / (1 - gamma)))
    n_rows = len(nS) * len(nA)
    n_cols = 3 if run_alternatives else 1
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols)
    t = 0
    for nstates in nS:
        for nactions in nA:
            mdp_distribution = MDPDist(n_states=nstates, n_actions=nactions, gamma=gamma)
            label = "Ns-" + str(nstates) + "-nA-" + str(nactions)
            ax = axs[t] if not run_alternatives else axs[t][0]
            policy = np.random.sample(size=(nstates, nactions))
            policy = policy / policy.sum(axis=1)[:, None]
            state = np.random.choice(nstates)
            values = []
            for i in range(n_samples):
                if i % 100 == 0:
                    print("Evaluating in %d model" % (i + 1))
                mdp = mdp_distribution.sample()
                value = mdp.policy_evaluation(policy=policy)
                v = value[state]
                values.append(v)
            sns.distplot(values, ax=ax, kde=True, label="current_state", color='c')
            if run_operator:
                if run_operator:
                    operator_values = []
                    for i in range(n_samples):
                        if i % 100 == 0:
                            print("Evaluating in %d model" % (i + 1))
                        mdp = mdp_distribution.sample()
                        value = mdp.policy_evaluation(policy=policy)
                        mdp.set_state(state)
                        a = np.random.choice(a=nactions, p=policy[state])
                        next_state, reward, done, info = mdp.step(a)
                        v = reward + gamma * value[next_state]
                        operator_values.append(v)
                    sns.distplot(operator_values, ax=ax, kde=True, label="target", color='r')
            if run_alternatives:
                values = []
                for i in range(n_samples):
                    if i % 100 == 0:
                        print("Evaluating in %d model" %(i+1))

                    mdp = mdp_distribution.sample()
                    s = state
                    ret = 0
                    for _ in range(episodes):
                        mdp.set_state(state)
                        for j in range(factor * effective_horizon):
                            a = np.random.choice(a=nactions, p=policy[s])
                            next_state, reward, done, info = mdp.step(a)
                            ret += reward * gamma ** j
                            mdp = mdp_distribution.sample()
                            mdp.set_state(state)
                    values.append(ret / episodes)
                ax = axs[t][1]
                sns.distplot(values, ax=ax, kde=True, label=label, color='c')
                values = []
                for i in range(n_samples):
                    if i % 100 == 0:
                        print("Evaluating in %d model" % (i + 1))
                    mdp = mdp_distribution.sample()
                    s = state
                    ret = 0
                    for _ in range(episodes):
                        mdp.set_state(state)
                        for j in range(factor * effective_horizon):
                            a = np.random.choice(a=nactions, p=policy[s])
                            next_state, reward, done, info = mdp.step(a)
                            ret += reward * gamma ** j
                            # mdp = mdp_distribution.sample()
                            # mdp.set_state(state)
                    values.append(ret / episodes)
                ax = axs[t][2]
                sns.distplot(values, ax=ax, kde=True, label=label, color='c')
            t += 1
    name = "epistemic_distributions_random_mdps_" + str(episodes) + "_" + str(n_samples)
    name += "op" if run_operator else ""
    plt.savefig(name + '.pdf')
    plt.show()


def run_parametric_mdps(mdp_distribution, n_states, n_actions, episodes=100, n_samples=2000, run_alternatives=False,
                        run_operator=False):
    effective_horizon = int(np.ceil(1 / (1 - gamma)))
    n_cols = 3 if run_alternatives else 1
    fig, axs = plt.subplots(nrows=1, ncols=n_cols)
    label = str(len(mdp_distribution.distribution)) + " mdps"
    ax = axs if not run_alternatives else axs[0]
    policy = np.random.sample(size=(n_states, n_actions))
    policy = policy / policy.sum(axis=1)[:, None]
    state = 0
    values = []
    for i in range(n_samples):
        if i % 100 == 0:
            print("Evaluating in %d model" % (i + 1))
        mdp = mdp_distribution.sample()
        value = mdp.policy_evaluation(policy=policy)
        v = value[state]
        values.append(v)
    sns.distplot(values, ax=ax, kde=True, label="current state", color='c')
    if run_operator:
        operator_values = []
        for i in range(n_samples):
            if i % 100 == 0:
                print("Evaluating in %d model" % (i + 1))
            mdp = mdp_distribution.sample()
            value = mdp.policy_evaluation(policy=policy)
            mdp.set_state(state)
            a = np.random.choice(a=n_actions, p=policy[state])
            next_state, reward, done, info = mdp.step(a)
            v = reward + gamma * value[next_state]
            operator_values.append(v)
        sns.distplot(operator_values, ax=ax, kde=True, label="target", color='r')
    if run_alternatives:
        values = []
        for i in range(n_samples):
            if i % 100 == 0:
                print("Evaluating in %d model" % (i + 1))

            mdp = mdp_distribution.sample()
            s = state
            ret = 0
            for ep in range(episodes):
                mdp.set_state(state)
                for j in range(factor * effective_horizon):
                    a = np.random.choice(a=n_actions, p=policy[s])
                    next_state, reward, done, info = mdp.step(a)
                    ret += reward * gamma ** j
                    mdp = mdp_distribution.sample()
                    mdp.set_state(state)
            values.append(ret / episodes)
        ax = axs[1]
        sns.distplot(values, ax=ax, kde=True, label=label, color='c')
        values = []
        for i in range(n_samples):
            if i % 100 == 0:
                print("Evaluating in %d model" % (i + 1))
            mdp = mdp_distribution.sample()
            s = state
            ret = 0
            for ep in range(episodes):
                mdp.set_state(state)
                for j in range(factor * effective_horizon):
                    a = np.random.choice(a=n_actions, p=policy[s])
                    next_state, reward, done, info = mdp.step(a)
                    ret += reward * gamma ** j
                    # mdp = mdp_distribution.sample()
                    # mdp.set_state(state)
            values.append(ret / episodes)
        ax = axs[2]
        sns.distplot(values, ax=ax, kde=True, label=label, color='c')
    name = "epistemic_distributions_chain_mdps_" + str(episodes) + "_" + str(n_samples)
    name += "op" if run_operator else ""
    plt.savefig(name +'.pdf')
    plt.show()


if __name__ == '__main__':
    nS = [10,100, 1000]
    nA = [2, 4, 8]
    nS = [10]
    nA = [2, 4]
    n_samples = 10000
    gamma = 0.99
    factor = 2
    random_mdps = True
    chain_states = 10
    episodes = 100
    run_alternatives = False
    run_operator = True
    if random_mdps:
        run_random_mdps(nS=nS, nA=nA, n_samples=n_samples, factor=factor, gamma=gamma, episodes=episodes,
                        run_alternatives=False, run_operator=True)
    else:
        dist = []
        slips = [0.5, 0.1, 0.01, 0.001, 0]
        weights = np.ones(len(slips)) / len(slips)
        for slip in slips:
            dist.append(generate_chain(chain_states, slip=slip, gamma=gamma))
        mdp_dist = MDPDist(n_states=chain_states, n_actions=2, distribution=dist, weights=weights, gamma=gamma,)
        run_parametric_mdps(mdp_distribution=mdp_dist, n_states=chain_states, n_actions=2, episodes=episodes,
                            n_samples=n_samples, run_alternatives=run_alternatives, run_operator=run_operator)

