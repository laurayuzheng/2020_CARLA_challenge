
import argparse
import json
import os
import sys
from time import strftime
from copy import deepcopy

from flow.core.util import ensure_dir
from flow.utils.registry import env_constructor
from flow.utils.rllib import FlowParamsEncoder, get_flow_params
from flow.utils.registry import make_create_env
import torch
from torch import det
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

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
        
    parser.add_argument(
        '--model-path', type=str, default=None,
        help='Path of model to load from. Should be in *.zip format.')

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
    elif name.lower() == "diff_td3":
        from diff_off_policy_algorithms.diff_td3 import DiffTD3
        kwargs = {
            'n_ext_steps': 1, 
            'sample_enhancement': None, 
            'se_n_augments': 1.0, 
            'se_augment_action_delta':1e-3, 
            'policy_enhancement': None
        }
        return DiffTD3, kwargs
    elif name.lower() == "diff_ddpg":
        from diff_off_policy_algorithms.diff_ddpg import DiffDDPG
        kwargs = {
            'n_ext_steps': 1, 
            'sample_enhancement': None, 
            'se_n_augments': 1.0, 
            'se_augment_action_delta':1e-3, 
            'policy_enhancement': None
        }
        return DiffDDPG, kwargs

def visualize_stablebaselines3(submodule, flags):
    flow_params = submodule.flow_params
    # Path to the saved files
    exp_tag = flow_params['exp_tag']
    algorithm_cls, kwargs = load_sb3_algorithm(flags.alg, flags)

    print('Loading the trained model and testing it out!')
    model = algorithm_cls.load(flags.model_path, device="cpu")
    flow_params['sim'].render = True
    env = env_constructor(params=flow_params, version=0)()
    # The algorithms require a vectorized environment to run
    eval_env = DummyVecEnv([lambda: env])
    obs = eval_env.reset()
    reward = 0
    for _ in range(flow_params['env'].horizon):
        # @BUGFIX: Have to set [deterministic] to be true to get real result.
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = eval_env.step(action)
        reward += rewards
    print('the final reward is {}'.format(reward))

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
        raise ValueError("Unable to find experiment config.")

    # Perform the training operation.
    visualize_stablebaselines3(submodule, flags)

if __name__ == "__main__":
    main(sys.argv[1:])