import os
import argparse
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import pandas as pd
import pickle

from tqdm import tqdm
from explore_latent_channel_strength import GradientSampler
from ganstudent_utils.latent_utils import S_LAYER_SIZES, LATENT_SPACE_DIM

import torch

import itertools
import copy

from torchvision import utils

import matplotlib.pyplot as plt
    
def parse_args():
    parser = argparse.ArgumentParser(description='Convert latents from W to S')

    parser.add_argument('--data-path', default='data/CelebA-HQ', help='path to dataset with latents dir')
    parser.add_argument('--boundary_path', help='Path to latent boundary')
    parser.add_argument('--latent_file_ext', default='.pickle', help='Format of latent code files')
    parser.add_argument('--flatten_s', action='store_true', help="Save a concatenated S code .npy rather than a layer dictionary.")

    parser.add_argument(
        "--name",
        type=str,
        default=str(datetime.now()),
        help="Name of this experiment",
    )

    parser.add_argument(
        "--out_dir",
        type=Path,
        default="outputs",
        help="path to generated outputs",
    )

    parser.add_argument(
        "--boundary_normal_path",
        type=Path,
        help="boundary.npy path file for latent space edit"
    )

    parser.add_argument(
        "--boundary_intercept_path",
        type=Path,
        help="intercept.npy path file for latent space edit"
    )

    parser.add_argument(
        "--max_latent_step",
        type=int,
        default=5,
        help="maximum latent space step"
    )

    parser.add_argument(
        "--number_edits",
        type=int,
        default=2,
        help="number of steps for each latent input"
    )

    parser.add_argument(
        "--edit_strategy",
        type=str,
        default='binary',
        help="editing strategy"
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="number of samples to be generated for each image",
    )

    parser.add_argument(
        "--size", type=int, default=1024, help="output image size of the generator"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="device to use, either cpu or cuda (cuda:0, cuda:1, ...)"
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default="stylegan2-ffhq-config-f.pt",
        help="path to the model checkpoint",
    )

    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")

    parser.add_argument(
        "--truncation_layer",
        type=int,
        default=None,
        help="Until what layer to apply truncation. If not specified, applies on all layers.",
    )

    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )

    parser.add_argument(
        "--optimize_steps",
        type=int,
        default=1,
        help="Number of gradient optimization steps per image pair",
    )

    parser.add_argument(
        "--latent_space",
        default='S',
        help="Latent space to investigate",
    )

    parser.add_argument("--map_layers", type=int, help="num of mapping layers", default=8)

    parser.add_argument("--max_dist_from_mean", type=float,
                        help="Maximum allowed distance of sampled w from the mean w", default=np.inf)

    parser.add_argument("--max_distance_from_plane", type=float,
                        help="Maximum allowed distance of sampled w from boundary", default=np.inf)    

    args = parser.parse_args()

    #TODO: convert these to proper args
    args.latent = 512
    args.n_mlp = args.map_layers

    return args

def get_latent_from_file(file_path):
    if str(file_path).endswith(".npy"):
        return np.expand_dims(np.load(file_path), axis=0)
    
    with open(file_path, 'rb') as fp:
        return pickle.load(fp)[0].unsqueeze(0).detach().cpu().numpy()

def save_s_code(file_path, code):
    if str(file_path).endswith(".npy"):
        np.save(file_path, np.concatenate([code[f'S{idx}'] for idx in range(26)], axis=1))
    else:
        with open(file_path, 'wb') as fp:
            pickle.dump(code, fp)

if __name__ == '__main__':
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    sampler = GradientSampler(args)

    latents_in_dir = os.path.join(args.data_path, 'latents')
    latents_out_dir = os.path.join(args.data_path, 's_latents')

    os.makedirs(latents_out_dir, exist_ok=True)
    output_ext = ".npy" if args.flatten_s else ".pickle"

    latent_file_names = [file_name for file_name in os.listdir(latents_in_dir) if file_name.endswith(args.latent_file_ext)]
    
    for file_name in tqdm(latent_file_names):
        try:
            latent_path = os.path.join(latents_in_dir, file_name)
            output_path = os.path.splitext(os.path.join(latents_out_dir, file_name))[0] + output_ext
            if os.path.exists(output_path):
                # To enable continuing a failed run
                continue

            latent = get_latent_from_file(latent_path)
            s_code = sampler.get_s_code_for_latent(latent)
            save_s_code(output_path, s_code)
        except Exception as e:
            print(f'Failed for {file_name} because {e}')

    if args.boundary_path:
        boundary = np.load(Path(args.boundary_path).joinpath('boundary.npy'))

        s_boundary = sampler.get_s_code_for_latent(boundary)

        boundary_out_path = Path(args.boundary_path).joinpath('s_boundary.pickle')
        save_s_code(boundary_out_path, s_boundary)

