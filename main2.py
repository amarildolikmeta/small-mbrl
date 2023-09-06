from src.training import MBRLLoop
from src.model import DirichletModel
from envs.env import setup_environment
import numpy as np
import hydra
import jax
import copy
import os


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--seed', type=int, default=0, help='Random seed')
#     parser.add_argument('--domain', type=str, default='mountain')
#     parser.add_argument('--dim', type=int, default=25)
#     parser.add_argument('--pac', action="store_true")
#     parser.add_argument('--ensemble', action="store_true")
#     parser.add_argument('--n_policies', type=int, default=1)
#     parser.add_argument('--gamma', type=float, default=0.99)
#     parser.add_argument('--alg', type=str, default='oac', choices=[
#         'oac', 'p-oac', 'sac', 'g-oac', 'g-tsac', 'p-tsac', 'ddpg', 'oac-w', 'gs-oac'
#     ])
#     parser.add_argument('--hockey_env', type=str, default='3dof-hit', choices=[
#         '3dof-hit', '3dof-defend', "7dof-hit", "7dof-defend", "7dof-prepare"
#     ])
#     parser.add_argument('--no_gpu', default=False, action='store_true')
#     parser.add_argument('--base_log_dir', type=str, default='./data')
#     parser.add_argument('--load_dir', type=str, default='')
#     parser.add_argument('--save_heatmap', action="store_true")
#     parser.add_argument('--comp_MADE', action="store_true")
#     parser.add_argument('--num_layers', type=int, default=2)
#     parser.add_argument('--layer_size', type=int, default=256)
#     parser.add_argument('--fake_policy', action="store_true")
#     parser.add_argument('--random_policy', action="store_true")
#     parser.add_argument('--expl_policy_std', type=float, default=0)
#     parser.add_argument('--target_paths_qty', type=float, default=0)
#     parser.add_argument('--dont_use_target_std', action="store_true")
#     parser.add_argument('--n_estimators', type=int, default=2)
#     parser.add_argument('--share_layers', action="store_true")
#     parser.add_argument('--mean_update', action="store_true")
#     parser.add_argument('--counts', action="store_true", help="count the samples in replay buffer")
#     parser.add_argument('--std_inc_prob', type=float, default=0.)
#     parser.add_argument('--prv_std_qty', type=float, default=0.)
#     parser.add_argument('--prv_std_weight', type=float, default=1.)
#     parser.add_argument('--std_inc_init', action="store_true")
#     parser.add_argument('--log_dir', type=str, default='./data')
#     parser.add_argument('--suffix', type=str, default='')
#     parser.add_argument('--max_path_length', type=int, default=1000)  # SAC: 1000
#     parser.add_argument('--replay_buffer_size', type=float, default=1e6)
#     parser.add_argument('--epochs', type=int, default=200)
#     parser.add_argument('--batch_size', type=int, default=256)  # SAC: 256
#     parser.add_argument('--r_min', type=float, default=0.)
#     parser.add_argument('--r_max', type=float, default=1.)
#     parser.add_argument('--r_mellow_max', type=float, default=1.)
#     parser.add_argument('--mellow_max', action="store_true")
#     parser.add_argument('--priority_sample', action="store_true")
#     parser.add_argument('--global_opt', action="store_true")
#     parser.add_argument('--save_sampled_data', default=False, action='store_true')
#     parser.add_argument('--n_components', type=int, default=1)
#     parser.add_argument('--snapshot_gap', type=int, default=10)
#     parser.add_argument('--keep_first', type=int, default=-1)
#     parser.add_argument('--save_fig', action='store_true')
#     parser.add_argument('--simple_reward', action='store_true')
#     parser.add_argument('--shaped_reward', action='store_true')
#     parser.add_argument('--jerk_only', action='store_true')
#     parser.add_argument('--high_level_action', action='store_true')
#     parser.add_argument('--delta_action', action='store_true')
#     parser.add_argument('--acceleration', action='store_true')
#     parser.add_argument('--include_joints', action='store_true')
#     parser.add_argument('--delta_ratio', type=float, default=0.1)
#     parser.add_argument('--max_accel', type=float, default=0.2)
#     parser.add_argument('--large_reward', type=float, default=1000)
#     parser.add_argument('--large_penalty', type=float, default=100)
#     parser.add_argument('--alpha_r', type=float, default=1.)
#     parser.add_argument('--c_r', type=float, default=0.)
#     parser.add_argument('--min_jerk', type=float, default=10000)
#     parser.add_argument('--max_jerk', type=float, default=100000)
#     parser.add_argument('--history', type=int, default=0)
#     parser.add_argument('--use_atacom', action='store_true')
#     parser.add_argument('--stop_after_hit', action='store_true')
#     parser.add_argument('--punish_jerk', action='store_true')
#     parser.add_argument('--parallel', type=int, default=1)
#     parser.add_argument('--interpolation_order', type=int, default=-1, choices=[1, 2, 3, 5, -1])
#     parser.add_argument('--include_old_action', action='store_true')
#     parser.add_argument('--use_aqp', action='store_true')
#     parser.add_argument('--n_threads', type=int, default=25)
#     parser.add_argument('--restore_only_policy', action='store_true')
#     parser.add_argument('--no_buffer_restore', action='store_true')
#     parser.add_argument('--speed_decay', type=float, default=0.5)
#     parser.add_argument('--aqp_terminates', action='store_true')
#     args = parser.parse_args()
#     return args

@hydra.main(config_path="config", config_name="run_test")
def experiment(args):

    env = setup_environment(
        args.env.env_setup,
        args.env.env_type,
        args.env.env_id,
        args.env.norm_reward,
        args.seed,
    )
    print(env)

    nState = env.nState
    nAction = env.nAction

    print(f'Training {args.train_type}')
    
    if hasattr(env, 'initial_distribution'):
        initial_distribution = env.initial_distribution
    else:
        initial_distribution = np.zeros(nState)
        initial_distribution[0] = 1.
    
    if hasattr(env, 'discount'):
        discount = env.discount
    else:
        discount = args.gamma

    print(args.train_type)
    data_dir = f'{args.env.env_id}_{args.train_type.type}_{args.optimization_type}' \
               f'_incorrectpriors{args.use_incorrect_priors}_clipped_bonus{args.clip_bonus}/'


    if args.use_softmax:
        data_dir += "/softmax_policy"
    elif args.use_logistic:
        data_dir += "/logistic_policy"
    else:
        raise ValueError("What policy should we use!")

    if args.const_lambda:
        data_dir += "/lambda_" + str(args.init_lambda) + "/"
    agent = DirichletModel(
        nState, 
        nAction, 
        int(args.seed),
        discount, 
        initial_distribution, 
        args.init_lambda,
        args.train_type.lambda_lr,
        args.train_type.policy_lr,
        args.use_incorrect_priors,
        use_softmax=args.use_softmax,
        use_logistic=args.use_logistic,
        use_adam=args.use_adam,
        clip_bonus=args.clip_bonus
    )
    
    p_params_baseline, baseline_true_perf = 0, 0
    if args.train_type.type in ["upper-cvar-opt-cvar"]:
        p_params_baseline, baseline_true_perf = get_baseline_policy(
            env, 
            args, 
            nState, 
            nAction, 
            discount, 
            initial_distribution
        )
    
    trainer = MBRLLoop(
        env, 
        agent, 
        nState, 
        nAction, 
        initial_distribution, 
        data_dir,
        p_params_baseline,
        baseline_true_perf,
        seed=int(args.seed),
        wandb_entity=args.wandb_entity,
        use_csv=args.csv_log,
    )

    if args.train_type.type == 'MC2PS': #  this one is only offline
        trainer.training_then_sample(args)
        return

    if args.train_type.type == "Q-learning":
        trainer.Q_learning(args, discount)
        return

    trainer.training_loop(args)


def get_baseline_policy(env, args, nState, nAction, discount, initial_distribution):
    #TODO:have to figure out how to do this with hydra
    #check if baseline file exists for env in directory
    filepath = f"baseline_policies_{args.env.env_id}.npy"
    if os.path.exists(filepath):
        # if yes, read file and return those params
        p_params_baseline = np.load(filepath)
        baseline_true_perf = np.load(f'{filepath[:-4]}_true_perf.npy')
    else:
        # if no, train policy to a mid-point on env, then save policy_params to file and return it as well
        # (can run a new agent and MBRLLoop just for PG or something?)
        agent = DirichletModel(
            nState, 
            nAction, 
            int(args.seed),
            discount, 
            initial_distribution, 
            args.init_lambda,
            args.train_type.lambda_lr,
            args.train_type.policy_lr,
            args.use_incorrect_priors,
            use_softmax=args.use_softmax,
            use_adam=args.use_adam
        )
        
        data_dir = f'baseline_{args.env.env_id}_{args.train_type.type}_{args.init_lambda}'

        trainer = MBRLLoop(
            env, 
            agent, 
            nState, 
            nAction, 
            initial_distribution, 
            data_dir,
            None,
            None
        )

        new_args = copy.deepcopy(args)
        new_args.num_eps = 200
        new_args.train_type.type = 'pg'
        new_args.train_type.policy_lr = 0.1
        new_args.train_type.mid_train_steps = 50

        print('training_loop')
        baseline_true_perf = trainer.training_loop(new_args)

        p_params_baseline = agent.policy.get_params()
        np.save(filepath, p_params_baseline)
        np.save(f'{filepath[:-4]}_true_perf.npy', baseline_true_perf)
    
    return p_params_baseline, baseline_true_perf


if __name__ == "__main__":
    jax.config.update('jax_platform_name', 'cpu')
    experiment()