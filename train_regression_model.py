import sys
import copy

sys.path.insert(0, '..')

import argparse
from pathlib import Path
from enum import Enum

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split

import matplotlib

import torch

import pickle
import os

matplotlib.use('Agg')

from utils.latent_utils import LatentCode, LatentSpace


class DistanceType(Enum):
    per_layer = 'per_layer'
    euclidean = 'euclidean'

    def __str__(self):
        return self.value

def convert_to_layer_data(data_frame, data_path, num_layers, boundary, distance_type,
                          file_suffix='.pickle', max_samples=None):
    data_per_layer = {}

    for layer_num in range(num_layers):
        pass

    # TODO: use pd.apply instead of iterating rows.
    for index, row in tqdm(data_frame.iterrows()):

        latent_path = Path(data_path).joinpath(str(index)).with_suffix(file_suffix)
        if not latent_path.exists():
            continue

        latent = LatentCode.from_file(latent_path)
        if distance_type == DistanceType.per_layer:
            distances = latent.layer_distance_to(boundary)
            num_distances = num_layers

        else:
            distances = [latent.distance_to(boundary)]
            num_distances = 1

        for layer_num in range(num_distances):
            data_per_layer.setdefault(layer_num, {'distances': [], 'gt': []})

            data_per_layer[layer_num]['distances'].append(distances[layer_num])
            data_per_layer[layer_num]['gt'].append(row[args.attribute])

        if max_samples and len(data_per_layer[layer_num]['distances']) > max_samples:
            break

    return data_per_layer


def filter_data(data_frame, data_path, file_suffix='.pickle', attribute=None):
    has_latents_list = []

    # TODO: use pd.apply instead of iterating rows.
    for index, row in tqdm(data_frame.iterrows()):

        latent_path = Path(data_path).joinpath(str(index)).with_suffix(file_suffix)

        if (attribute and not attribute in row) or (not latent_path.exists()):
            continue

        has_latents_list.append(index)

    return data_frame.loc[has_latents_list]


def prepare_features_and_gt_pairs(fit_layers, data_per_layer, per_layer_weights=None):
    feature_maps = np.concatenate(
        [np.expand_dims(data_per_layer[layer_idx]['distances'], axis=1) for layer_idx in fit_layers],
        axis=1)

    if per_layer_weights is not None:
        feature_maps = np.expand_dims(feature_maps @ per_layer_weights, axis=1)

    gt = np.array(data_per_layer[0]['gt'])

    return feature_maps, gt


def sample_min_distance(all_data, points_to_sample, min_distance=3):
    # NOTE: this is a very confusing code, sorry! The try_sample_idx is meaningless, it's just a way to keep sampling
    # withing the range of possible values. The actual_idx is the idx in the data.
    data = copy.deepcopy(all_data)
    options = np.arange(data.shape[0])
    sampled_points = []
    sampled_idx = []

    while options.size > 0 and len(sampled_points) < points_to_sample:
        try_sample_idx = np.random.choice(options.size, size=1)

        actual_idx = options[try_sample_idx]
        try_sample_feature = all_data[actual_idx]
        if len(sampled_points) == 0 or \
                np.linalg.norm(np.array(sampled_points) - try_sample_feature, axis=1).min() > min_distance:
            sampled_points.append(try_sample_feature)
            sampled_idx.append(actual_idx.item())

        options = np.delete(options, try_sample_idx)

    if len(sampled_points) != points_to_sample:
        print(f'WARNING: failed sampling for n={points_to_sample} and min_d={min_distance}.'
              f' Sampled {len(sampled_idx)} points instead')

    return sampled_idx


def scatter_2d(name, x, y):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    model = LinearRegression()
    reg = model.fit(x, y)
    score = reg.score(x, y)
    print(f'{name}:{score}')
    sns.set(rc={'figure.figsize': (20, 10)})
    df = pd.DataFrame({'distance': np.squeeze(x), 'gt': np.squeeze(y)})

    plot = sns.regplot(data=df, x='distance', y='gt')
    plt.title(
        f'Age as function of our features, R^2={score}')
    fig = plot.get_figure()
    fig.tight_layout()
    fig.savefig(f'scatter_{name}_features_and_gt.png')
    plt.clf()


def resolve_feature_min_distance(all_data, user_value, data_size, num_points):
    if user_value is not None:
        if user_value < 0:
            user_value = -np.inf

        return user_value

    value_range = all_data.max() - all_data.min()
    formula = value_range / (num_points ** 1.3)
    return formula


def parse_args():
    parser = argparse.ArgumentParser(description='Linear regressor training script.')

    parser.add_argument('--data-path', required=True, help='path to dataset')
    parser.add_argument('--output-dir', required=True, help='path to output dir')

    parser.add_argument('--annotations-file', default='data/celeba_yaw_look_right_no_dist.csv',
                        help='path to annotation file')
    parser.add_argument('--boundary_path', required=True,
                        help='Path to latent boundary')
    parser.add_argument('--attribute', type=str, required=True,
                        help='Which attribute of the CelebA to use')

    parser.add_argument('--layer_weights_path', type=str,
                        help="Path to layer weights numpy file. If provided, will collapse layer distances to one value using these weights.")

    parser.add_argument('--normalize_layer_weights', action='store_true', help="Rescale layer weights so they sum to 1")

    parser.add_argument('--boundary_to_wp', action='store_true',
                        help='Convert a provided W boundary to W+ by repeating.')

    parser.add_argument('--fit_layers', nargs='+', type=int,
                        help='Limit features to use distances only from the given layer numbers.')

    parser.add_argument('--latent_file_ext', default=".pickle", choices=[".pickle", ".pkl", ".npy"],
                        help="Extension of latent code files")
    parser.add_argument('--boundary_file_ext', default=".npy", choices=[".pickle", ".pkl", ".npy"],
                        help="Extension of boundary files")

    parser.add_argument('--regularization', nargs='+', type=str,
                        help='What types of regularization to apply on regression')

    parser.add_argument('--train_size', type=int, default=1000)

    parser.add_argument('--labeled_feature_min_distance', type=float, help='Negative numbers for no min distance')
    parser.add_argument('--feature_sample_ratio', type=float, default=1)

    parser.add_argument('--distance_type', type=DistanceType, choices=list(DistanceType),
                        default=DistanceType.per_layer,
                        help='How to calculate distance between latent code and boundary')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(2)
    np.random.seed(2)

    boundary = LatentCode.from_file(Path(args.boundary_path).joinpath("boundary").with_suffix(args.boundary_file_ext))

    if args.boundary_to_wp:
        boundary = boundary.to_wp()

    if boundary.latent_space is LatentSpace.S:
        num_layers = 26
    elif boundary.latent_space is LatentSpace.WP:
        num_layers = boundary.num_style_layers
    else:
        num_layers = 1

    if args.fit_layers:
        fit_layers = args.fit_layers
    elif args.distance_type == DistanceType.euclidean:
        fit_layers = [0]
    else:
        fit_layers = list(range(num_layers))

    if args.layer_weights_path:
        layer_weights = np.load(args.layer_weights_path)
    elif args.distance_type == DistanceType.euclidean:
        layer_weights = np.ones(num_layers)
    else:
        layer_weights = None

    if layer_weights is not None:
        layer_weights = layer_weights[fit_layers]

    if layer_weights is not None and args.normalize_layer_weights:
        layer_weights /= np.sum(layer_weights)

    print("Preparing data...")
    df = pd.read_csv(args.annotations_file, delim_whitespace=True, header=0)

    df = filter_data(df, args.data_path, file_suffix=args.latent_file_ext, attribute=args.attribute)

    train_df, test_df = train_test_split(df, train_size=args.train_size)

    data_per_layer_train = convert_to_layer_data(train_df, args.data_path, num_layers, boundary, args.distance_type,
                                                 file_suffix=args.latent_file_ext)
    data_per_layer_test = convert_to_layer_data(test_df, args.data_path, num_layers, boundary, args.distance_type,
                                                file_suffix=args.latent_file_ext)

    train_feature_maps, train_gt = prepare_features_and_gt_pairs(fit_layers, data_per_layer_train, layer_weights)
    test_feature_maps, test_gt = prepare_features_and_gt_pairs(fit_layers, data_per_layer_test, layer_weights)

    print(f"All data loaded. Total samples: {train_feature_maps.shape[0]}")

    if args.feature_sample_ratio != 1:
        if train_feature_maps.shape[1] != 1:
            raise NotImplementedError('Point sampling is only implemented for single-dimensional features.')

        low_half_tail = (1 - args.feature_sample_ratio) / 2
        high_half_tail = args.feature_sample_ratio + low_half_tail

        max_feature = np.quantile(train_feature_maps, high_half_tail)
        min_feature = np.quantile(train_feature_maps, low_half_tail)

        in_range_idxs = (train_feature_maps.squeeze() < max_feature) & (train_feature_maps.squeeze() > min_feature)

        train_feature_maps = train_feature_maps[in_range_idxs]
        train_gt = train_gt[in_range_idxs]

    print(f'Taking just the {args.feature_sample_ratio * 100}% center data.'
          f' Samples remaining: {train_feature_maps.shape[0]}')

    print(f"Working in latent space: {boundary.latent_space}")
    print(f"Using layers: {fit_layers}")

    error_by_points = []
    for num_points in [2, 5, 10, 20, 1000]:
        error = {'MAE': [], 'R2': [], 'coefs': []}

        idxs = []

        if num_points == args.train_size:
            num_iters = 1
            min_distance = 0
        else:
            num_iters = 1000
            min_distance = resolve_feature_min_distance(train_feature_maps, args.labeled_feature_min_distance,
                                                        args.train_size, num_points)

        for n in tqdm(range(num_iters)):
            selected_idxs = sample_min_distance(train_feature_maps, num_points, min_distance)
            idxs.extend([sorted(selected_idxs)])
            selected_distance = train_feature_maps[selected_idxs]
            selected_gt = train_gt[selected_idxs]

            if args.regularization is None:
                model = LinearRegression()
            elif args.regularization == ['L1']:
                model = Lasso(max_iter=10000)
            elif args.regularization == ['L2']:
                model = Ridge()
            elif set(args.regularization) == {'L2', 'L1'}:
                model = ElasticNet(max_iter=1000000)
            else:
                raise NotImplementedError(f'Regularization {args.regularization} is not implemented.')

            reg = model.fit(selected_distance, selected_gt)

            score = reg.score(test_feature_maps, test_gt)

            predict = reg.predict(test_feature_maps)
            mae = np.mean(np.abs(predict - test_gt))

            error['MAE'].append(mae)
            error['R2'].append(score)
            error['coefs'].append(reg.coef_)

            error_by_points.append({'num_points': num_points, 'MAE': mae, 'R2': score, 'method': 'Ours'})

        try:
            unique, counts = np.unique(np.array(idxs), return_counts=True, axis=0)
            print(f'For n={num_points}: there were {unique.shape[0]} unique groups sampled')

        except Exception as e:
            pass

        mMAE = np.mean(error['MAE'])
        stdMAE = np.std(error['MAE'])
        maxMAE = np.max(error['MAE'])

        mR2 = np.mean(error['R2'])
        stdR2 = np.std(error['R2'])

        text = f'For n={num_points}, with min distance: {min_distance}. MAE - {mMAE:.3} +- {stdMAE:.3} ({maxMAE:.3})'

        print(text)

        with open(os.path.join(args.output_dir, f"model_{num_points}.pickle"), 'wb') as fp:
            pickle.dump(reg, fp)

