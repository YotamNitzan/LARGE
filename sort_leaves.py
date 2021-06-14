import sys
import copy

sys.path.insert(0, '..')

import argparse
from pathlib import Path
import os

import numpy as np

from tqdm import tqdm

import matplotlib

import torch
import shutil

import cv2

matplotlib.use('Agg')

from ganstudent_utils.latent_utils import LatentCode, LatentSpace


# TODO: This file copies extensively from latent_svm.py and should be cleaned / changed to avoid repetitions.

# Example args for sorting:
# sort.py --latent_path /disk2/rinong/data/GANStudent/CelebA-HQ-inverted/psp/val/inference_results/s_latents/ 
#         --image_path /disk2/rinong/data/GANStudent/CelebA-HQ/val/ 
#         --boundary_path /disk2/rinong/output/GANStudent/ffhq_clip_features/double_trait_boundaries/makeup_15/
#         --num_samples 10
#         --out_dir /disk2/rinong/output/GANStudent/sorted/
#         --seed 12
#         --num_for_plot 10

def parse_args():
    parser = argparse.ArgumentParser(description='SVM Training')

    parser.add_argument('--latent_path', default='data/CelebA-HQ', help='path to dataset')
    parser.add_argument('--image_path', default='data/CelebA-HQ', help='path to dataset')

    parser.add_argument('--boundary_path', required=True, help='Path to latent boundary')

    parser.add_argument('--negative_boundary_path', help='Path to negative latent boundary')

    parser.add_argument('--latent_file_ext', default=".pickle", choices=[".pickle", ".pkl", ".npy"],
                        help="Extension of latent code files")

    parser.add_argument('--boundary_file_ext', default=".pickle", choices=[".pickle", ".pkl", ".npy"],
                        help="Extension of boundary files")

    parser.add_argument('--num_samples', type=int, default=10, help="Number of images to sample from directory")

    parser.add_argument('--num_for_plot', type=int, default=10, help="Number of images to use in sorting plot")

    parser.add_argument('--out_dir', required=True, help="Path to output directory")

    parser.add_argument('--layer_weights_path', type=str,
                        help="Path to layer weights numpy file. If provided, will collapse layer distances to one value using these weights.")

    parser.add_argument('--boundary_to_wp', action='store_true',
                        help='Convert a provided W boundary to W+ by repeating.')

    parser.add_argument('--seed', type=int, default=42, help="Random seed")

    parser.add_argument('--resolution', type=int, default=1024, help="StyleGAN reso")

    args = parser.parse_args()

    return args


def get_latent_img_pairs(latent_dir, image_dir, num_samples=None, latent_file_ext=".pickle"):
    latent_files = np.array([os.path.join(latent_dir, file_name) for file_name in os.listdir(latent_dir) if
                             file_name.endswith(latent_file_ext)])
    latent_files = [x for x in latent_files if ('Early_blight' in x) or ('healthy' in x)]
    # latent_files = [x for x in latent_files if ('greening' not in x)]

    np.random.shuffle(latent_files)

    file_pairs = []
    healthy = 0
    sick = 0
    for latent_path in latent_files:

        file_idx = os.path.basename(latent_path).split(".")[0]
        img_path = os.path.join(image_dir, file_idx + ".jpg")

        if not os.path.isfile(img_path):
            continue

        # file_pairs.append({"img": img_path, "latent": latent_path})

        if ('healthy' in img_path and healthy < num_samples // 2 - 1):
            file_pairs.append({"img": img_path, "latent": latent_path})
            healthy += 1

        if ('healthy' not in img_path):
            file_pairs.append({"img": img_path, "latent": latent_path})
            sick += 1

        if num_samples and len(file_pairs) >= num_samples:
            break

    return file_pairs


if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    boundary = LatentCode.from_file(Path(args.boundary_path).joinpath("boundary").with_suffix(args.boundary_file_ext))
    try:
        intercept = np.load(Path(args.boundary_path).joinpath("intercept").with_suffix(args.boundary_file_ext)).item()
    except Exception as e:
        intercept = 0

    negative_boundary = LatentCode.from_file(
        Path(args.negative_boundary_path).joinpath("boundary").with_suffix(args.boundary_file_ext)) \
        if args.negative_boundary_path else None

    num_wp_styles = int(2 * np.log2(args.resolution) - 2)

    if args.boundary_to_wp:
        boundary = boundary.to_wp(num_wp_styles)

        if negative_boundary is not None:
            negative_boundary = negative_boundary.to_wp(num_wp_styles)

    if boundary.latent_space is LatentSpace.S:
        num_layers = 26
    elif boundary.latent_space is LatentSpace.WP:
        num_layers = num_wp_styles
    else:
        num_layers = 1

    fit_layers = list(range(num_layers))

    layer_weights = np.load(args.layer_weights_path) if args.layer_weights_path else None
    layer_weights /= np.sum(layer_weights)

    for run_idx in range(1):
        file_pairs = get_latent_img_pairs(args.latent_path, args.image_path, args.num_samples, args.latent_file_ext)

        print(f"Sorting {len(file_pairs)} files...")

        for pair in file_pairs:
            pair_latent = LatentCode.from_file(pair['latent'])

            # boundary_dist = pair_latent.distance_to(boundary)
            #
            # if negative_boundary is not None:
            #     boundary_dist -= pair_latent.distance_to(negative_boundary)
            #
            # pair['distance'] = boundary_dist

            feature_maps = np.concatenate(
                [np.expand_dims(pair_latent.layer_distance_to(boundary, intercept)[layer_idx], axis=0) for layer_idx in fit_layers],
                axis=0)

            if layer_weights is not None:
                pair['distance'] = feature_maps @ layer_weights


        sorted_pairs = sorted(file_pairs, key=lambda x: x['distance'])
        classification = [x['img'].split('/')[-1].split('_')[-1] for x in sorted_pairs]
        sorted_classification = sorted(classification)
        if sorted_classification != classification and sorted_classification != list(reversed(classification)):
            continue

        # for idx, pair in enumerate(sorted_pairs):
        #     dst_img = os.path.join(args.out_dir, f'{idx:04d}' + ".jpg")
        #     shutil.copy2(pair['img'], dst_img)

        sample_idxes = np.linspace(0, len(sorted_pairs) -1 , args.num_for_plot)
        # sample_idxes = range(0, len(sorted_pairs), args.num_for_plot)

        sorted_img = np.concatenate([cv2.imread(sorted_pairs[int(idx)]['img']) for idx in sample_idxes], axis=1)
        cv2.imwrite(os.path.join(args.out_dir, f"all_sorted_{run_idx}.png"), sorted_img)

        with open(os.path.join(args.out_dir, f"all_sorted_{run_idx}.txt"), 'w') as fp:
            for i in sample_idxes:
                x = sorted_pairs[int(i)]
                name=x['img'].split('/')[-1]
                distance = x['distance']
                fp.write(f'{name}={distance:.3f}\n')