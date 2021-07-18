import pickle
import numpy as np

import torch

from enum import Enum

LATENT_SPACE_DIM = 512

POSSIBLE_NUM_STYLES = [14, 16, 18]
S_LAYER_SIZES = [512, ] * 15 + [256, ] * 3 + [128, ] * 3 + [64, ] * 3 + [32, ] * 2
S_WP_SIZES = [512, ] * 2 + [512, 512 * 2, ] * 4 + [512, 256 * 2, ] + [256, 128 * 2, ] + [128, 64 * 2, ] + [64, 32 * 2]


class LatentSpace(Enum):
    W = 0,
    WP = 1,
    S = 2


def is_sorted(arr):
    return np.all(np.sort(arr) == arr) or np.all(np.sort(arr)[::-1] == arr)


def load_latent_pkl(file_path):
    with open(file_path, 'rb') as fp:
        latent = pickle.load(fp)

    return latent


def projection_distance(code, boundary, intercept=0):
    return code.dot(boundary.T) + intercept


class LatentCode(object):

    def __init__(self, code, space: LatentSpace):
        self.latent_space = space
        self.code = code

        self.code_as_layers = self._code_as_layers()

    def _code_as_layers(self):
        return NotImplementedError

    def distance_to(self, boundary: 'LatentCode', intercept=0):
        assert self.latent_space == boundary.latent_space

        return projection_distance(self.code, boundary.get_code_as_array(), intercept)

    def layer_distance_to(self, boundary: 'LatentCode', intercept=0):
        assert self.latent_space == boundary.latent_space

        layer_distances = []
        for code_layer, boundary_layer in zip(self.code_as_layers, boundary.get_code_as_layers()):
            layer_distances.append(projection_distance(code_layer, boundary_layer, intercept))

        if len(layer_distances) == 1:
            return layer_distances[0]

        return layer_distances

    def get_code_as_array(self):
        return self.code

    def get_code_as_layers(self):
        return self.code_as_layers

    def to(self, dest_latent_space):
        if self.latent_space == dest_latent_space or self.latent_space == LatentSpace[dest_latent_space]:
            return self
        elif self.latent_space == LatentSpace.W and (
                dest_latent_space == LatentSpace.WP or LatentSpace[dest_latent_space] == LatentSpace.WP):
            return self.to_wp()
        elif self.latent_space == LatentSpace.WP and (
                dest_latent_space == LatentSpace.W or LatentSpace[dest_latent_space] == LatentSpace.W):
            return self.to_w()
        else:
            raise NotImplementedError(
                f'Conversion between {self.latent_space} and {dest_latent_space} is not implemented')

    @staticmethod
    def from_file(file_path):
        file_path = str(file_path)

        if file_path.endswith(".npy"):
            latent = np.load(file_path)
        elif file_path.endswith(".pkl") or file_path.endswith(".pickle"):
            latent = load_latent_pkl(file_path)
            if isinstance(latent, torch.Tensor):
                latent = latent.detach().cpu().numpy()
        else:
            raise NotImplementedError(f'Cannot load latent with suffix {file_path.suffix}')

        if len(latent) == 1:
            latent = latent[0]

        if len(latent) == LATENT_SPACE_DIM:
            return WLatentCode(latent)

        for num_styles in POSSIBLE_NUM_STYLES:
            if len(latent) == num_styles or len(latent) == num_styles * LATENT_SPACE_DIM:
                latent = np.reshape(latent, [-1])
                return WPLatentCode(latent, num_styles)

        if len(latent) == np.sum(S_LAYER_SIZES):
            return SLatentCode(latent)

        if len(latent) == len(S_LAYER_SIZES):
            if isinstance(latent, dict):
                latent = np.concatenate([latent[f'S{i}'] for i in range(len(S_LAYER_SIZES))], axis=1)[0]
            elif isinstance(latent, list):
                latent = np.concatenate(latent, axis=0)
            else:
                raise NotImplementedError("S latents only support dict or list formats.")
            return SLatentCode(latent)

        raise ValueError(f'Could not resolve type of latent in {file_path}')


class WLatentCode(LatentCode):

    def __init__(self, code):
        super(WLatentCode, self).__init__(code, LatentSpace.W)

    def _code_as_layers(self):
        return np.reshape(self.code, [1, 512])

    def to_wp(self, num_layers=18):
        return WPLatentCode(np.tile(self.code, num_layers), num_style_layers=num_layers)


class WPLatentCode(LatentCode):
    def __init__(self, code, num_style_layers=18):
        self.num_style_layers = num_style_layers
        super(WPLatentCode, self).__init__(code, LatentSpace.WP)

    def _code_as_layers(self):
        return np.reshape(self.code, [self.num_style_layers, 512])

    def to_w(self):
        _, idxs, counts = np.unique(self.code_as_layers, axis=0, return_counts=True, return_inverse=True)
        if len(counts) == 1 or (len(counts) == 2 and is_sorted(idxs)):
            # Sometimes we truncate only in a the first few layers, so there will be 2 unique values - appearing
            # one after the other
            return WLatentCode(self.code_as_layers[0])
        else:
            raise NotImplementedError(f'This W+ code is not a W vector, translation is not defined')


class SLatentCode(LatentCode):
    def __init__(self, code):
        super(SLatentCode, self).__init__(code, LatentSpace.S)

    def _code_as_layers(self):
        latent_list = []

        curr_idx = 0
        for idx, size in enumerate(S_LAYER_SIZES):
            latent_list.append(self.code[curr_idx:curr_idx + size])
            curr_idx += size

        return latent_list
