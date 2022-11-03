import argparse
import os
import sys

import jax.random as jr
from revisiting_svae import dict_product, run_pendulum

if __name__ == '__main__':
    
    # Create the parser
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--inference_method', type=str)

    parser.add_argument('--latent_dims', type=int)
    parser.add_argument('--rnn_dims', type=int)
    parser.add_argument('--seed', type=int)
    
    parser.add_argument('--dataset_size', type=str)
    parser.add_argument('--snr', type=str)

    parser.add_argument('--base_lr', type=float)
    parser.add_argument('--prior_base_lr', type=float)
    
    parser.add_argument('--use_natural_grad', type=bool)
    parser.add_argument('--constrain_prior', type=bool)

    parser.add_argument('--lr_decay', type=bool)
    parser.add_argument('--prior_lr_warmup', type=bool)
    parser.add_argument('--group_tag', type=str)

    parser.add_argument('--mask_size', type=int)
    parser.add_argument('--mask_start', type=int)
    
    # run_params = {
    #     # Most important: inference method
    #     "inference_method": "svae",
    #     # Relevant dimensions
    #     "latent_dims": 2,
    #     "rnn_dims": 10,
    #     "seed": jr.PRNGKey(0),
    #     "dataset_size": "medium", # "small", "medium", "large"
    #     "snr": "medium", # "small", "medium", "large"
    #     "use_natural_grad": False,
    #     "constrain_prior": False,
    #     "base_lr": 1e-2,
    #     "prior_base_lr": 1e-2,
    #     "prior_lr_warmup": True,
    #     "lr_decay": False,
    #     "group_tag": "test_prediction",
    #     # The only pendulum-specific entry, will be overridden by params expander
    #     "mask_size": 40,
    #     # "plot_interval": 1,
    #     "mask_start": 0,#1000
    # }
    args = parser.parse_args()
    run_params = vars(args)
    run_params["seed"] = jr.PRNGKey(run_params["seed"])

    run_pendulum(run_params)