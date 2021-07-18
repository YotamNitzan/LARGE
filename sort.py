import sys

sys.path.insert(0, '..')

import os
import copy
import shutil
import argparse
from enum import Enum
from pathlib import Path

import cv2
import torch
import matplotlib

matplotlib.use('Agg')
import numpy as np
from tqdm import tqdm

from utils.latent_utils import LatentCode, LatentSpace


class DistanceType(Enum):
    per_layer = 'per_layer'
    euclidean = 'euclidean'

    def __str__(self):
        return self.value


def parse_args():
    parser = argparse.ArgumentParser(description='Sort script')

    parser.add_argument('--latent_path', default='data/CelebA-HQ', help='path to dataset')
    parser.add_argument('--image_path', default='data/CelebA-HQ', help='path to dataset')

    parser.add_argument('--boundary_path', required=True, help='Path to latent boundary')
    parser.add_argument('--negative_boundary_path', help='Path to negative latent boundary')

    parser.add_argument('--latent_file_ext', default=".pickle", choices=[".pickle", ".pkl", ".npy"],
                        help="Extension of latent code files")

    parser.add_argument('--num_samples', type=int, default=10, help="Number of images to sample from directory")
    parser.add_argument('--balanced_classes', type=str, nargs='+', help="Filter data to this classes and balance them")

    parser.add_argument('--num_for_plot', type=int, default=10, help="Number of images to use in sorting plot")

    parser.add_argument('--out_dir', required=True, help="Path to output directory")

    parser.add_argument('--layer_weights_path', type=str,
                        help="Path to layer weights numpy file."
                             " If provided, will collapse layer distances to one value using these weights.")

    parser.add_argument('--boundary_to_wp', action='store_true',
                        help='Convert a provided W boundary to W+ by repeating.')

    parser.add_argument('--seed', type=int, default=42, help="Random seed")

    parser.add_argument('--model_layers', type=int, default=18, help="Number of W+ layers for the given model.")
    parser.add_argument('--distance_type', type=DistanceType, choices=list(DistanceType),
                        default=DistanceType.per_layer,
                        help='How to calculate distance between latent code and boundary')

    parser.add_argument('--resize_output', nargs='+', type=int)
    parser.add_argument('--weighted_edges', default=False, choices=[True, False], type=bool,
                        help='Over-weight edges of sort for datasets with small std')

    args = parser.parse_args()

    return args


def find_class_of_file(filename, classes):
    for class_name in classes:
        if class_name in filename:
            return class_name
    return None


def get_latent_img_pairs(latent_dir, image_dir, num_samples=None, latent_file_ext=".pickle",
                         balanced_classes_names=None):
    latent_files = np.array([os.path.join(latent_dir, file_name) for file_name in os.listdir(latent_dir) if
                             file_name.endswith(latent_file_ext)])

    if balanced_classes_names:
        latent_files_filtered = []
        for latent_file in latent_files:
            class_name = find_class_of_file(latent_file, balanced_classes_names)
            if class_name:
                latent_files_filtered.append(latent_file)

        latent_files = latent_files_filtered

    np.random.shuffle(latent_files)

    file_pairs = []

    class_counts = {}

    # TODO: use pd.apply instead of iterating rows.
    for latent_path in latent_files:

        file_idx = os.path.basename(latent_path).split(".")[0]
        img_path = os.path.join(image_dir, file_idx + ".jpg")

        if not os.path.isfile(img_path):
            continue

        if balanced_classes_names:
            class_name = find_class_of_file(latent_path, balanced_classes_names)
            c = class_counts.setdefault(class_name, 0)
            if c >= num_samples // len(balanced_classes_names):
                # Too many of this class
                continue
            else:
                class_counts[class_name] = c + 1

        file_pairs.append({"img": img_path, "latent": latent_path})

        if num_samples and len(file_pairs) >= num_samples:
            break

    return file_pairs


if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    boundary = LatentCode.from_file(args.boundary_path)

    negative_boundary = LatentCode.from_file(args.negative_boundary_path) \
        if args.negative_boundary_path else None

    if args.boundary_to_wp:
        boundary = boundary.to_wp(args.model_layers)

        if negative_boundary is not None:
            negative_boundary = negative_boundary.to_wp(args.model_layers)

    if boundary.latent_space is LatentSpace.S:
        num_layers = 26
    elif boundary.latent_space is LatentSpace.WP:
        num_layers = args.model_layers
    else:
        num_layers = 1

    fit_layers = list(range(num_layers))

    layer_weights = np.load(args.layer_weights_path) if args.layer_weights_path else None

    file_pairs = get_latent_img_pairs(args.latent_path, args.image_path, args.num_samples, args.latent_file_ext,
                                      args.balanced_classes)

    print(f"Sorting {len(file_pairs)} files...")

    for pair in tqdm(file_pairs):
        pair_latent = LatentCode.from_file(pair['latent'])

        if args.distance_type == DistanceType.euclidean:
            distance = pair_latent.distance_to(boundary)
            if negative_boundary is not None:
                distance -= pair_latent.distance_to(negative_boundary)
        else:
            distances = np.concatenate(
                [np.expand_dims(pair_latent.layer_distance_to(boundary)[layer_idx], axis=0)
                 for layer_idx in fit_layers], axis=0)

            if layer_weights is not None:
                distance = distances @ layer_weights
            else:
                distance = np.mean(distances)

        pair['distance'] = distance

    sorted_pairs = sorted(file_pairs, key=lambda x: x['distance'])

    for idx, pair in enumerate(sorted_pairs):
        dst_img = os.path.join(args.out_dir, f'{idx:04d}' + ".jpg")
        shutil.copy2(pair['img'], dst_img)

    if args.num_for_plot < args.num_samples and args.weighted_edges:
        rel_num = args.num_for_plot / 3
        sample_idxes = np.concatenate([
            np.linspace(0, args.num_samples // 10, int(np.floor(rel_num))),
            np.linspace(args.num_samples // 10 + 1, 9 * args.num_samples // 10 - 1, int(np.ceil(rel_num))),
            np.linspace(9 * args.num_samples // 10, args.num_samples - 1, int(np.floor(rel_num))),
        ])
    else:
        sample_idxes = np.linspace(0, args.num_samples - 1, args.num_for_plot)

    imgs = [cv2.imread(sorted_pairs[int(idx)]['img']) for idx in sample_idxes]

    if args.resize_output:
        imgs = [cv2.resize(x, tuple(args.resize_output)) for x in imgs]

    sorted_img = np.concatenate(imgs, axis=1)
    cv2.imwrite(os.path.join(args.out_dir, "all_sorted.jpg"), sorted_img)
