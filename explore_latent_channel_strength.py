import os
import argparse
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
import torch

import itertools
import copy

from torchvision import utils

import matplotlib.pyplot as plt

LATENT_SPACE_DIM = 512

S_LAYER_SIZES = [512, ] * 15 + [256, ] * 3 + [128, ] * 3 + [64, ] * 3 + [32, ] * 2
S_WP_SIZES = [512, ] * 2 + [512, 512 * 2, ] * 4 + [512, 256 * 2, ] + [256, 128 * 2, ] + [128, 64 * 2, ] + [64, 32 * 2]

import torch
import numpy as np

from models.stylegan2 import Generator

class LatentSampler(object):

    _LATENT_SPACE_DIM = 512

    def __init__(self, args):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.g_ema = Generator(
            args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
        ).to(self.device)
        checkpoint = torch.load(args.ckpt, map_location=self.device)

        self.g_ema.load_state_dict(checkpoint["g_ema"], strict=False)
        self.mean_latent = self.g_ema.mean_latent(4096).cpu().detach().numpy()

    def sample(self, num_samples):
        z_codes = torch.randn(num_samples, LatentSampler._LATENT_SPACE_DIM, device=self.device)
        w_codes = self.g_ema.get_latents([z_codes])[0].cpu().detach().numpy()

        return w_codes

def project_latent_to_plane(latent, boundary, intercept):
    # https://stackoverflow.com/questions/9605556/how-to-project-a-point-onto-a-plane-in-3d
    dist = latent.dot(boundary.T) + intercept
    proj_latent = latent - dist * boundary
    return proj_latent, np.squeeze(np.abs(dist))

class GradientSampler(LatentSampler):
    def __init__(self, args):
        super(GradientSampler, self).__init__(args)

        self.mean_latent = torch.tensor(self.mean_latent, device=self.device)

        self.boundary = np.load(args.boundary_normal_path)
        self.intercept = np.load(args.boundary_intercept_path) if args.boundary_intercept_path else np.zeros(1)

        self.max_latent_step = args.max_latent_step
        self.number_edits = args.number_edits

        self.comparison_loss = get_loss()

        self.init_gradient_accumulators()

        self.register_gradient_hooks()

    def edit_latents(self, latent):
        # Ensure we get meaningful differences in images by lower-bounding the distance + editing a pair in both directions (positive / negative).
        shift = np.random.uniform(self.max_latent_step / 2, self.max_latent_step, (np.shape(latent)[0], 2, 1))
        shift[:, 1, :] *= -1.

        latent_spectrum = np.expand_dims(latent, axis=1) + shift * self.boundary

        return latent_spectrum

    def generate_from_latents(self, latents):
        images, _ = self.g_ema(
            [latents], truncation=0.5, randomize_noise=False,
            truncation_latent=self.mean_latent, input_is_latent=True)

        return images

    def generate_from_s(self, s_code):
        self.overwrite_s_code(s_code)

        images, _ = self.g_ema(
            [torch.randn(s_code['S0'].size()[0], LATENT_SPACE_DIM, device=self.device)], truncation=0.5,
            randomize_noise=False,
            truncation_latent=self.mean_latent, input_is_latent=True)

        return images

    def register_layer_forward_backward_hook(self, layer, name):
        layer.register_forward_hook(get_activation(self.activation_dict, name))
        layer.register_backward_hook(get_gradient(self.gradient_dict, name))

    def register_gradient_hooks(self):
        self.gradient_dict = {}
        self.activation_dict = {}

        self.register_layer_forward_backward_hook(self.g_ema.conv1.conv.modulation, f'S0')
        self.register_layer_forward_backward_hook(self.g_ema.to_rgb1.conv.modulation, f'S1')

        layers = int(np.log2(self.g_ema.size) - 2)
        for layer_idx in range(layers):
            conv_idx = layer_idx * 2
            s_idx = 3 * layer_idx + 2
            for sub_layer_idx in range(2):
                self.register_layer_forward_backward_hook(self.g_ema.convs[conv_idx + sub_layer_idx].conv.modulation,
                                                          f'S{s_idx + sub_layer_idx}')

            self.register_layer_forward_backward_hook(self.g_ema.to_rgbs[layer_idx].conv.modulation, f'S{s_idx + 2}')

    def overwrite_s_code(self, s_code):

        self.g_ema.conv1.conv.modulation.register_forward_hook(set_activation(s_code, f'S0'))
        self.g_ema.to_rgb1.conv.modulation.register_forward_hook(set_activation(s_code, f'S1'))

        layers = int(np.log2(self.g_ema.size) - 2)
        for layer_idx in range(layers):
            conv_idx = layer_idx * 2
            s_idx = 3 * layer_idx + 2
            for sub_layer_idx in range(2):
                self.g_ema.convs[conv_idx + sub_layer_idx].conv.modulation.register_forward_hook(
                    set_activation(s_code, f'S{s_idx + sub_layer_idx}'))

            self.g_ema.to_rgbs[layer_idx].conv.modulation.register_forward_hook(set_activation(s_code, f'S{s_idx + 2}'))

    def get_s_code_for_latent(self, latent_code):
        self.generate_from_latents(torch.tensor(latent_code, device=self.device))

        return copy.deepcopy(self.activation_dict)

    def init_gradient_accumulators(self, latents=['W', 'W+', 'S']):
        # For general image-editing directions
        self.gradient_accumulators = {latent: 0. for latent in latents}
        self.optimization_accumulators = {latent: 0. for latent in latents}
        self.first_step_grads = {latent: 0. for latent in latents}

        # For computing normalization with random samples
        self.gradient_accumulators_random = {latent: 0. for latent in latents}

    def get_latent_gradients(self, latent_space, latents):
        if latent_space != "S":
            return latents.grad.data.cpu().numpy()
        else:
            return np.concatenate([latents[f'S{idx}'].grad.data.cpu().numpy() for idx in range(26)], axis=1)

    def accumulate_gradients(self, latents, latent_space, src_images, dst_images, optimizer):
        self.g_ema.zero_grad()
        optimizer.zero_grad()

        loss = self.comparison_loss(src_images, dst_images)
        loss.backward(retain_graph=True)

        self.optimization_accumulators[latent_space] += self.get_latent_gradients(latent_space, latents)
        return loss

    def save_image(self, image_tensor, output_dir, file_name):
        image_path = os.path.join(output_dir, file_name)

        utils.save_image(
            image_tensor[0],
            image_path,
            nrow=1,
            normalize=True,
            range=(-1, 1),
        )

    def get_wp_for_latent(self, latent):
        num_layers = int(2 * np.log2(self.g_ema.size) - 2)
        return np.repeat(np.expand_dims(latent, axis=1), num_layers, 1)

    def latent_convert_func(self, latent_space):
        if latent_space == 'W':
            return np.identity
        if latent_space == 'W+':
            return self.get_wp_for_latent
        if latent_space == 'S':
            return self.get_s_code_for_latent

        raise NotImplementedError(f"No latent conversion function defined for chosen latent space: {latent_space}.")

    def image_sampling_func(self, latent_space):
        if latent_space in ['W', 'W+']:
            return self.generate_from_latents
        if latent_space == 'S':
            return self.generate_from_s

        raise NotImplementedError(f"No image sampling function defined for chosen latent space: {latent_space}.")

    def latents_to_tensors(self, latent_space, src_latents, dst_latents):
        if latent_space in ['W', 'W+']:
            src_latent_tensor = torch.nn.Parameter(torch.tensor(src_latents, requires_grad=True, device=self.device))
            dst_latent_tensor = torch.tensor(dst_latents, device=self.device)

            return src_latent_tensor, dst_latent_tensor

        if latent_space == 'S':
            src_latent_tensor = {key: torch.nn.Parameter(torch.tensor(value, requires_grad=True, device=self.device))
                                 for key, value in src_latents.items()}
            dst_latent_tensor = {key: torch.tensor(value, device=self.device) for key, value in dst_latents.items()}

            return src_latent_tensor, dst_latent_tensor

        raise NotImplementedError(f"No tensor-conversion function defined for chosen latent space: {latent_space}.")

    def latent_optimization_distance(self, latent_space, src_latent_tensor, dst_latent_tensor):
        if latent_space in ['W', 'W+']:
            w_gap = np.mean(np.abs(src_latent_tensor.cpu().detach().numpy() -
                                   dst_latent_tensor.cpu().numpy()))
            return w_gap

        if latent_space == 'S':
            flattened_src = np.concatenate([src_latent_tensor[f'S{idx}'].cpu().detach().numpy() for idx in range(26)],
                                           axis=1)
            flattened_dst = np.concatenate([dst_latent_tensor[f'S{idx}'].cpu().numpy() for idx in range(26)], axis=1)

            s_gap = np.mean(np.abs(flattened_src - flattened_dst))
            return s_gap

        raise NotImplementedError(
            f"No latent optimization distance function defined for chosen latent space: {latent_space}.")

    def sample_gradients(self, optimize_steps=1, output_dir=None, latent_space='W', sample_with_edit=True):

        if sample_with_edit:
            latents = self.sample(1)
            latents, _ = project_latent_to_plane(latents, self.boundary, self.intercept)
            latents = self.edit_latents(latents).astype(np.float32)

            grad_accumulator = self.gradient_accumulators
        else:
            latents = np.expand_dims(self.sample(2), 0)
            grad_accumulator = self.gradient_accumulators_random

        src_latents = latents[:, 0]
        dst_latents = latents[:, 1]

        src_latents = self.latent_convert_func(latent_space)(src_latents)
        dst_latents = self.latent_convert_func(latent_space)(dst_latents)

        src_latent_tensor, dst_latent_tensor = self.latents_to_tensors(latent_space, src_latents, dst_latents)

        params = list(src_latent_tensor.values()) if isinstance(src_latent_tensor, dict) else [src_latent_tensor]
        opt = torch.optim.Adam(params, lr=1e-2)

        dst_images = self.image_sampling_func(latent_space)(dst_latent_tensor)

        for i in range(optimize_steps):
            src_images = self.image_sampling_func(latent_space)(src_latent_tensor)

            loss = self.accumulate_gradients(src_latent_tensor, latent_space, src_images, dst_images, opt)

            if i == 0:

                if output_dir:
                    self.save_image(dst_images, output_dir, "dst.png")

                self.first_step_grads[latent_space] += np.abs(self.optimization_accumulators[latent_space].copy())

            if i % 100 == 0 or i == (optimize_steps - 1):
                latent_distance = self.latent_optimization_distance(latent_space, src_latent_tensor, dst_latent_tensor)

                tqdm.write(f"[{i + 1}/{optimize_steps}] - Latent distance: {latent_distance}, loss: {loss}")
                self.save_image(src_images, output_dir, f"src_{str(i).zfill(6)}.png")

            opt.step()

        grad_accumulator[latent_space] += np.abs(self.optimization_accumulators[latent_space])
        self.optimization_accumulators[latent_space] = 0.


def get_loss(use_l1=True):
    return (
        torch.nn.L1Loss()
        if use_l1
        else torch.nn.MSELoss()
    )


def get_activation(activation_dict, name):
    def hook(model, input, output):
        activation_dict[name] = output.detach().cpu().numpy()

    return hook


def get_gradient(gradient_dict, name):
    def hook(model, input, output):
        if (len(output) > 1):
            print(len(output))
            for item in output:
                print(item.size())
            exit(0)

        gradient_dict[name] = output[0].detach().cpu().numpy()

    return hook


def set_activation(activation_dict, name):
    def hook(model, input, output):
        return activation_dict[name]

    return hook


def s_grads_to_layer(s_grads, layer_sizes):
    boundary = 0
    grads_by_layer = []
    for size in layer_sizes:
        grads_by_layer.append(np.mean(s_grads[boundary:boundary + size]))
        boundary += size

    return grads_by_layer


def plot_latent_directions(latent_strengths, output_dir, file_name):
    plt.plot(np.arange(0, len(latent_strengths)), latent_strengths / np.max(latent_strengths))
    plt.savefig(os.path.join(output_dir, file_name))
    plt.clf()


def get_main_directions(latent_strengths, num_dir):
    return sorted(np.argpartition(latent_strengths, -num_dir)[-num_dir:])


def explore_latents(args):

    torch.manual_seed(8)
    np.random.seed(8)

    args.out_dir.mkdir(exist_ok=True, parents=True)

    sampler = GradientSampler(args)

    print(f"Exploring latent gradient strengths for latent space {args.latent_space}")

    for i in tqdm(range(args.random_samples)):
        # random images require considerably more optimization steps to align.
        sampler.sample_gradients(optimize_steps=args.random_optimize_steps, output_dir=args.out_dir,
                                 latent_space=args.latent_space, sample_with_edit=False)

    for i in tqdm(range(args.edit_samples)):
        sampler.sample_gradients(optimize_steps=args.optimize_steps, output_dir=args.out_dir,
                                 latent_space=args.latent_space)

    grad_norm = np.squeeze(sampler.gradient_accumulators_random[args.latent_space] / args.random_samples)

    grads = np.squeeze(sampler.gradient_accumulators[args.latent_space] / args.edit_samples) / grad_norm
    boundary = np.squeeze(np.abs(sampler.boundary))
    first_step_grads = np.squeeze(sampler.first_step_grads[args.latent_space] / args.edit_samples)

    for var, name in [(first_step_grads, "first_grads"),
                      (grads, "grads"),
                      (boundary, "boundary"),
                      (grad_norm, "random_grads")]:

        plot_latent_directions(var.reshape(-1), args.out_dir, name + ".png")
        print(f'Main directions for {name}: {get_main_directions(var.reshape(-1), 10)}')

        if name != "boundary" and args.latent_space == "W+":
            layer_vars = np.mean(var, axis=1)
            plot_latent_directions(layer_vars, args.out_dir, name + "_layers.png")
            print(f'Main layers for {name}: {get_main_directions(layer_vars, 4)}')

        if name != "boundary" and args.latent_space == "S":
            layer_vars = s_grads_to_layer(var, S_LAYER_SIZES)
            plot_latent_directions(layer_vars, args.out_dir, name + "_s_layers.png")
            print(f'Main s layers for {name}: {get_main_directions(layer_vars, 4)}')

            layer_vars = s_grads_to_layer(var, S_WP_SIZES)
            plot_latent_directions(layer_vars, args.out_dir, name + "_s_wp_layers.png")
            print(f'Main wp layers for {name}: {get_main_directions(layer_vars, 4)}')

    weighted_layers = s_grads_to_layer(grads, S_LAYER_SIZES) if args.latent_space == "S" else np.mean(grads, axis=1)
    weighted_layers /= np.sum(grads)
    np.save(os.path.join(args.out_dir, f"{args.latent_space}_layer_weights.npy"), weighted_layers)
    np.save(os.path.join(args.out_dir, f"{args.latent_space}_channel_weights.npy"), grads / np.sum(grads))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create generated & edited dataset")

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
        "--random_samples",
        type=int,
        default=100,
        help="number of samples to be generated for gradient normalization",
    )

    parser.add_argument(
        "--edit_samples",
        type=int,
        default=100,
        help="number of samples to be generated for editing direction steps",
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
        help="Number of gradient optimization steps per edited image pair",
    )

    parser.add_argument(
        "--random_optimize_steps",
        type=int,
        default=1,
        help="Number of gradient optimization steps per random image pair",
    )
    parser.add_argument(
        "--latent_space",
        default='W+',
        help="Latent space to investigate",
    )

    parser.add_argument("--map_layers", type=int, help="num of mapping layers", default=8)

    parser.add_argument("--max_dist_from_mean", type=float,
                        help="Maximum allowed distance of sampled w from the mean w", default=np.inf)

    parser.add_argument("--max_distance_from_plane", type=float,
                        help="Maximum allowed distance of sampled w from boundary", default=np.inf)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    # TODO: convert these to proper args
    args.latent = 512
    args.n_mlp = args.map_layers

    explore_latents(args)
