import argparse
import os
import sys
from pprint import pprint
import jax.random as jr
from revisiting_svae import dict_product, run_nlb

if __name__ == '__main__':
    
    # Create the parser
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--inference_method', type=str)
    parser.add_argument('--run_type', type=str)

    parser.add_argument('--latent_dims', type=int)
    parser.add_argument('--seed', type=int)
    
    parser.add_argument('--dataset_size', type=str)
    
    parser.add_argument('--base_lr', type=float)
    parser.add_argument('--prior_base_lr', type=float)
    
    parser.add_argument('--sample_kl', type=int)
    parser.add_argument('--constrain_prior', type=int)
    parser.add_argument('--constrain_dynamics', type=int)

    parser.add_argument('--lr_decay', type=int)
    parser.add_argument('--prior_lr_warmup', type=int)
    parser.add_argument('--group_tag', type=str)

    parser.add_argument('--conv_kernel_size', type=str)

    parser.add_argument('--beta_schedule', type=str)

    args = parser.parse_args()
    run_params = vars(args)
    run_params["seed"] = jr.PRNGKey(run_params["seed"])
    run_nlb(run_params)