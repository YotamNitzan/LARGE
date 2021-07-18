import os
import pickle
import argparse

import numpy as np

from train_regression_model import DistanceType
from utils.latent_utils import LatentCode, LatentSpace

def parse_args():
    parser = argparse.ArgumentParser(description='Linear regressor inference script.')

    parser.add_argument('--pretrained_model', required=True, help='path to pretrained regression model')

    parser.add_argument('--boundary_path', required=True, help='Path to latent boundary')
    parser.add_argument('--latent_path', required=True, help='Path to latent code file')

    parser.add_argument('--layer_weights_path', type=str, 
                        help="Path to layer weights numpy file. If provided, will collapse layer distances to one value using these weights.")

    parser.add_argument('--normalize_layer_weights', action='store_true', help="Rescale layer weights so they sum to 1")

    parser.add_argument('--boundary_to_wp', action='store_true',
                        help='Convert a provided W boundary to W+ by repeating.')

    parser.add_argument('--fit_layers', nargs='+', type=int,
                        help='Limit features to use distances only from the given layer numbers.')

    parser.add_argument('--distance_type', type=DistanceType, choices=list(DistanceType),
                        default=DistanceType.per_layer,
                        help='How to calculate distance between latent code and boundary')

    args = parser.parse_args()

    return args

def get_latent_distances(args, boundary):

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

    latent = LatentCode.from_file(args.latent_path)

    if args.distance_type == DistanceType.per_layer:
        distances = latent.layer_distance_to(boundary)
    else:
        distances = [latent.distance_to(boundary)]

    distances = np.expand_dims(distances, axis=0)
    distances = distances[:, fit_layers]
    

    if layer_weights is not None:
        distances = distances @ np.expand_dims(layer_weights, axis=1)

    return distances

if __name__ == "__main__":
    args = parse_args()


    # load regression model
    print(f"Using regression model checkpoint from {args.pretrained_model}...")
    with open(args.pretrained_model, 'rb') as fp:
        model = pickle.load(fp)

    # prepare latent boundary
    print(f"Loading boundary from {args.boundary_path}...")
    boundary = LatentCode.from_file(args.boundary_path)

    if args.boundary_to_wp:
        boundary = boundary.to_wp()

    distances = get_latent_distances(args, boundary)

    print(f"Predicted score for atttribute: {model.predict(distances)[0]:.2f}")



    