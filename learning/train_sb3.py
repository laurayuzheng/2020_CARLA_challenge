"""Runner script for single and multi-agent reinforcement learning experiments.

This script performs an RL experiment using the PPO algorithm. Choice of
hyperparameters can be seen and adjusted from the code below. 

It only uses Stable Baselines 3 to train the model.

Usage
    python learning/train_sb3.py d_singleagent_figure_eight diff_ddpg --exp_name 0  --pe --se 
    python learning/train_sb3.py d_singleagent_merge ppo --exp-name testNewSUMO
    OMP_NUM_THREADS=1 nohup python -u learning/train_sb3.py d_singleagent_figure_eight ddpg --exp_name exp5 > train_baseddpg_figeight5.txt &
"""
import argparse
import json
import os
import sys
from time import strftime
from copy import deepcopy
import gym

from flow.core.util import ensure_dir
from flow.utils.registry import env_constructor
from flow.utils.rllib import FlowParamsEncoder, get_flow_params
from flow.utils.registry import make_create_env
import torch
from torch import det
import numpy as np

def parse_args(args):
    """Parse training options user can specify in command line.

    Returns
    -------
    argparse.Namespace
        the output parser object
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Parse argument used when running a Flow simulation.",
        epilog="python train.py EXP_CONFIG")

    # required input parameters
    parser.add_argument(
        'exp_config', type=str,
        help='Name of the experiment configuration file, as located in '
             'exp_configs/rl/singleagent or exp_configs/rl/multiagent.')
    parser.add_argument(
        'alg', type=str,
        help='Name of learning algorithm to import from SB3.')

    # optional input parameters
    parser.add_argument(
        '--num_steps', type=int, default=1000000, # 1,000,000 retrain
        help='How many total steps to perform learning over')
    parser.add_argument(
        '--num_eval_steps', type=int, default=5000, # 200 evaluations = 1 mil / 5000 
        help='How many steps to train before a single evaluation')
    parser.add_argument(
        '--rollout_size', type=int, default=1024,
        help='How many steps are in a training batch.')
    parser.add_argument(
        '--checkpoint_path', type=str, default=None,
        help='Directory with checkpoint to restore training from.')
    parser.add_argument(
        '--exp-name', type=str, default="test",
        help='Optional experiment name to tag results. ')
    parser.add_argument(
        '--se', action='store_true', help='Sample enhancement.')
    parser.add_argument(
        '--pe', action='store_true', help='Policy enhancement.')
    parser.add_argument(
        '--train-diff', action='store_true', help='Train diff version of algorithm alongside baseline experiment.')
    parser.add_argument(
        '--init-path', type=str, default=None, help='Define folder containing "init_model.zip" to initialize model training with. ')
    parser.add_argument(
        '--max-perturb', type=float, default=2e-1, help='Maximum threshold for policy enhancement of on-policy algorithms. Relevant for: DiffPPO, DiffTRPO')

# n_ext_steps=1, sample_enhancement=args.se, se_n_augments=1.0, se_augment_action_delta=1e-3, policy_enhancement=args.pe
    return parser.parse_known_args(args)[0]


def load_sb3_algorithm(name, flags):
    kwargs = {}

    if name.lower() == "ppo":
        from stable_baselines3 import PPO
        kwargs["n_steps"] = flags.rollout_size 
        return PPO, kwargs
    elif name.lower() == "sac": 
        from stable_baselines3 import SAC 
        return SAC, kwargs
    elif name.lower() == "ddpg":
        from stable_baselines3 import DDPG 
        return DDPG, kwargs
    elif name.lower() == "td3":
        from stable_baselines3 import TD3 
        return TD3, kwargs
    elif name.lower() == "a2c":
        from stable_baselines3 import A2C 
        return A2C, kwargs
    elif name.lower() == "trpo": 
        from sb3_contrib import TRPO 
        return TRPO, kwargs 
    elif name.lower() == "diff_td3":
        from diff_off_policy_algorithms.diff_td3 import DiffTD3
        kwargs = {
            'n_ext_steps': 1, 
            'sample_enhancement': flags.se, 
            'se_n_augments': 1.0, 
            'se_augment_action_delta':1e-3, 
            'policy_enhancement': flags.pe
        }
        return DiffTD3, kwargs
    elif name.lower() == "diff_ddpg":
        from diff_off_policy_algorithms.diff_ddpg import DiffDDPG
        kwargs = {
            'n_ext_steps': 1, 
            'sample_enhancement': flags.se, 
            'se_n_augments': 1.0, 
            'se_augment_action_delta':1e-3, 
            'policy_enhancement': flags.pe
        }
        return DiffDDPG, kwargs
    elif name.lower() == "diff_sac":
        from diff_off_policy_algorithms.diff_sac import DiffSAC
        kwargs = {
            'n_ext_steps': 1, 
            'sample_enhancement': flags.se, 
            'se_n_augments': 1.0, 
            'se_augment_action_delta':1e-3, 
            'policy_enhancement': flags.pe
        }
        return DiffSAC, kwargs
    elif name.lower() == "diff_ppo":
        from diff_on_policy_algorithms.diff_ppo import DiffPPO
        kwargs = {
            'sample_enhancement': flags.se, 
            'policy_enhancement': flags.pe, 
            'max_perturb': flags.max_perturb
        }
        return DiffPPO, kwargs
    elif name.lower() == "diff_trpo":
        from diff_on_policy_algorithms.diff_trpo import DiffTRPO
        kwargs = {
            'sample_enhancement': flags.se, 
            'policy_enhancement': flags.pe, 
            'max_perturb': flags.max_perturb
        }
        return DiffTRPO, kwargs

def run_model_stablebaseline(flow_params,
                             alg_cls,
                             num_steps=50,
                             num_eval_steps=50,
                             eval_log_path=None, 
                             init_model_path="",
                             **alg_kwargs,):
    """Run the model for num_steps if provided.

    Parameters
    ----------
    flow_params : dict
        flow-specific parameters
    num_cpus : int
        number of CPUs used during training
    rollout_size : int
        length of a single rollout
    num_steps : int
        total number of training steps
    The total rollout length is rollout_size.

    Returns
    -------
    stable_baselines.*
        the trained model
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
    # from stable_baselines3 import PPO

    constructor, env_name = env_constructor(params=flow_params, version=0)

    eval_flow_params = flow_params.copy() 

    if "eval_env_name" in flow_params:
        eval_flow_params["env_name"] = flow_params["eval_env_name"]

    constructor2, env_name2 = env_constructor(params=eval_flow_params, version=0)

    # print(flow_params["env_name"])
    # print(eval_flow_params["env_name"])

    constructor_train = constructor() 
    constructor_eval = constructor2()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu' 

    # The algorithms require a vectorized environment to run
    env = DummyVecEnv([lambda: constructor_train])
    eval_env = VecMonitor(DummyVecEnv([lambda: constructor_eval]), eval_log_path)
    
    if "ddpg" in alg_cls.__name__.lower() or "td3" in alg_cls.__name__.lower():
        from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        alg_kwargs["action_noise"] = action_noise

    policy = "MlpPolicy" 

    if "diff_trpo" in alg_cls.__name__.lower() or "diff_ppo" in alg_cls.__name__.lower(): 
        from diff_on_policy_algorithms.diff_policies import DiffActorCriticPolicy
        policy = DiffActorCriticPolicy

    train_model = alg_cls(policy, env, verbose=1, device=device, **alg_kwargs)

    # If a model initialization path is specified
    if init_model_path:
        train_model.set_parameters(os.path.join(init_model_path, "init_model"))

    # Save the current model's initialization
    train_model.save(os.path.join(eval_log_path, "init_model"))

    train_model.learn(total_timesteps=num_steps, eval_env=eval_env, eval_freq=num_eval_steps, n_eval_episodes=4, eval_log_path=eval_log_path)
    return train_model

def train_stable_baselines(submodule, flags):
    """Train policies using the PPO algorithm in stable-baselines."""
    from stable_baselines3.common.vec_env import DummyVecEnv
    # from stable_baselines3 import PPO

    flow_params = submodule.flow_params
    # Path to the saved files
    exp_tag = flow_params['exp_tag']
    result_name = '{}/{}/{}_{}'.format(exp_tag, flags.alg, flags.exp_name, strftime("%Y-%m-%d-%H:%M:%S"))

    path = os.path.realpath(os.path.expanduser('./results'))
    save_path = os.path.join(path, result_name)
    ensure_dir(save_path)

    with open(save_path + "/setting.txt", 'w') as f:
        f.write("Algorithm: {} / Num Steps: {} / Num Eval Steps: {} / Rollout Size: {}".format(
            flags.alg, flags.num_steps, flags.num_eval_steps, flags.rollout_size))
    
    algorithm_cls, kwargs = load_sb3_algorithm(flags.alg, flags)

    # Perform training.
    print('Beginning training.')
    model = run_model_stablebaseline(
        flow_params, algorithm_cls, flags.num_steps, flags.num_eval_steps, save_path, flags.init_path, **kwargs)

    # Save the model to a desired folder and then delete it to demonstrate
    # loading.
    print('Saving the trained model!')
    train_model_path = os.path.join(save_path, "model")
    model.save(train_model_path)

    # dump the flow params
    with open(train_model_path + '.json', 'w') as outfile:
        json.dump(flow_params, outfile,
                  cls=FlowParamsEncoder, sort_keys=True, indent=4)

    # Replay the result by loading the model
    # print('Loading the trained model and testing it out!')
    # model = algorithm_cls.load(train_model_path, device="cpu")
    # flow_params = get_flow_params(train_model_path + '.json')
    # flow_params['sim'].render = True
    # constructor, _ = env_constructor(params=flow_params, version=0)
    # env = constructor()
    # # The algorithms require a vectorized environment to run
    # eval_env = DummyVecEnv([lambda: env])
    # obs = eval_env.reset()
    # reward = 0
    # for _ in range(flow_params['env'].horizon):
    #     # @BUGFIX: Have to set [deterministic] to be true to get real result.
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, rewards, dones, info = eval_env.step(action)
    #     reward += rewards
    # print('the final reward is {}'.format(reward))
    return save_path


def main(args):
    """Perform the training operations."""
    # Parse script-level arguments (not including package arguments).
    flags = parse_args(args)

    # Import relevant information from the exp_config script.
    module = __import__(
        "exp_configs.rl.singleagent", fromlist=[flags.exp_config])
    module_d = __import__(
        "exp_configs.rl.d_singleagent", fromlist=[flags.exp_config])
    module_ma = __import__(
        "exp_configs.rl.multiagent", fromlist=[flags.exp_config])

    # Import the sub-module containing the specified exp_config and determine
    # whether the environment is single agent or multi-agent.
    if hasattr(module, flags.exp_config):
        submodule = getattr(module, flags.exp_config)
        multiagent = False
    elif hasattr(module_d, flags.exp_config):
        submodule = getattr(module_d, flags.exp_config)
        multiagent = False
    elif hasattr(module_ma, flags.exp_config):
        submodule = getattr(module_ma, flags.exp_config)
        assert flags.rl_trainer.lower() in ["rllib", "h-baselines"], \
            "Currently, multiagent experiments are only supported through "\
            "RLlib. Try running this experiment using RLlib: " \
            "'python train.py EXP_CONFIG'"
        multiagent = True
    else:
        raise ValueError("Unable to find experiment config: ", flags.exp_config)

    # Just for training purposes 
    thresholds = [4e-1, 2e-1, 1e-1]
    exp_names = ["thresh04", "thresh02", "thresh01"]

    # Perform the training operation.
    print("Training model on algorithm ", flags.alg)

    if flags.alg == "diff_trpo" or flags.alg == "diff_ppo":
        for i in range(len(thresholds)):
            flags.max_perturb = thresholds[i]
            flags.exp_name = exp_names[i]
            print("Training for perturb value: ", flags.max_perturb)
            print("Experiment name: ", flags.exp_name)
            print("Initializing with weights: ", flags.init_path)
            exp_path = train_stable_baselines(submodule, flags)
    else:
        exp_path = train_stable_baselines(submodule, flags)
        
    # If we want to train diff algorithm alongside baseline
    if flags.train_diff and "diff_" not in flags.alg: # Make sure it's not already a diff algorithm
        print("Training model on diff version of algorithm ", flags.alg)
        flags.alg = "diff_"+flags.alg 
        flags.init_path = exp_path 
        
        if not flags.se or not flags.pe: 
            print("Warning: Training diff algorithm is enabled but SE, PE, or both flags are false. ")
        
        if not flags.se and not flags.pe: 
            flags.se = True 
            flags.pe = True 
        
        # diff_exp_path = train_stable_baselines(submodule, flags)

        if "ppo" in flags.alg or "trpo" in flags.alg:
            for i in range(len(thresholds)):
                flags.max_perturb = thresholds[i]
                flags.exp_name = exp_names[i]
                print("Training for perturb value: ", flags.max_perturb)
                print("Initializing with weights: ", flags.init_path)
                diff_exp_path = train_stable_baselines(submodule, flags)
        else:
            diff_exp_path = train_stable_baselines(submodule, flags)

        # print("\nBaseline model path: %s\nDiff model path:%s\n" % (exp_path, diff_exp_path))

    print("Done.")

if __name__ == "__main__":
    main(sys.argv[1:])