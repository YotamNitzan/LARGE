import clip
import torch
import os

import numpy as np

from tqdm import tqdm

import re

import matplotlib.pyplot as plt
import argparse

seed_rgx  = re.compile(r"(seed)(\d+)_")
layer_rgx = re.compile(r"(layer)(\d+)_")
code_rgx  = re.compile(r"(code)(\d+)_")
delta_rgx = re.compile(r"(delta)(-?\d+\.\d)")

imagenet_templates = [
    'a bad photo of a {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

def get_textual_direction(class_names, templates, clip_model):
    with torch.no_grad():
        textual_features = []
        for class_name in class_names:
            text = [template.format(class_name) for template in templates]
            text = clip.tokenize(text).cuda()
            class_text_embedding = clip_model.encode_text(text)

            # normalize
            class_text_embedding /= class_text_embedding.norm(dim=-1, keepdim=True)
            class_text_embedding = class_text_embedding.mean(dim=0)
            class_text_embedding /= class_text_embedding.norm()

            textual_features.append(class_text_embedding)

    textual_feature_tensor = torch.stack(textual_features, dim=1).t()
    
    text_direction = (textual_feature_tensor[0] - textual_feature_tensor[1]).cpu().numpy()
    text_direction /= np.linalg.norm(text_direction)

    return text_direction

def get_boundary_for_direction(latent_codes, clip_embeddings, text_direction, percentile=80):

    num_codes = len(latent_codes)

    weighted_codes = []

    idxes = np.arange(num_codes)
    np.random.shuffle(idxes)
    
    src_idx = idxes[:num_codes // 2]
    dst_idx = idxes[num_codes // 2:]

    delta_codes = []
    projections = []
    for i, j in tqdm(zip(src_idx, dst_idx)):
        delta_code = latent_codes[i] - latent_codes[j]

        delta_embeddings = clip_embeddings[i] - clip_embeddings[j]
        delta_embeddings /= np.linalg.norm(delta_embeddings)

        projection = delta_embeddings.dot(text_direction)

        delta_codes.append(delta_code)
        projections.append(projection)

    projections = np.array(projections)
    min_projection = np.percentile(np.abs(projections), percentile)

    projections[np.abs(projections) < min_projection] = 0.0

    weighted_codes = np.array(delta_codes) * np.expand_dims(projections, 1)

    weighted_codes = np.mean(weighted_codes, axis=0)
    weighted_codes /= np.linalg.norm(weighted_codes)
    
    return weighted_codes

def load_latents_and_features(data_dir):
    latents_dir = os.path.join(data_dir, 'latents')
    feature_dir = os.path.join(data_dir, 'clip_features')

    latent_dir_list = os.listdir(latents_dir)

    latent_files = [os.path.join(latents_dir, file_name) for file_name in tqdm(latent_dir_list) if file_name.endswith(".npy")]
    feature_files = [os.path.join(feature_dir, file_name) for file_name in tqdm(latent_dir_list) if file_name.endswith(".npy")]

    latents = np.concatenate([np.load(latent_file) for latent_file in tqdm(latent_files)], axis=0)
    features = np.concatenate([np.load(feature_file) for feature_file in tqdm(feature_files)], axis=0)

    return latents, features

def get_val_by_regex(string, regex, type):
    try:
        type(regex.findall(string)[0][1])
    except Exception as e:
        print(string)
        raise e
    return type(regex.findall(string)[0][1])

def precompute_feature_directions(data_dir, model_layers):

    latents_dir = os.path.join(data_dir, 'latents')
    feature_dir = os.path.join(data_dir, 'clip_features')

    latent_dir_list = os.listdir(latents_dir)

    latent_files = [os.path.join(latents_dir, file_name) for file_name in tqdm(latent_dir_list) if file_name.endswith(".npy") and 'layer' in file_name]
    feature_files = [os.path.join(feature_dir, file_name) for file_name in tqdm(latent_dir_list) if file_name.endswith(".npy") and 'layer' in file_name]

    feature_dir_array = np.zeros(shape=(model_layers, 512, 2, 512))

    for feature_file in tqdm(feature_files):
        layer_num = get_val_by_regex(feature_file, layer_rgx, int)
        code_idx = get_val_by_regex(feature_file, code_rgx, int)
        delta = get_val_by_regex(feature_file, delta_rgx, float)
        delta_direction = delta > 0.
    
        feature_dir_array[layer_num, code_idx, int(delta_direction)] += np.load(feature_file)[0]

    feature_dir_array /= 100.0
    feature_dir_array = feature_dir_array[:, :, 1, :] - feature_dir_array[:, :, 0, :]

    return feature_dir_array

def get_boundary_for_direction_precomp(clip_embeddings, text_direction, model_layers, percentile=80):

    projections = np.zeros((model_layers, 512))
    
    for layer_idx in tqdm(range(model_layers)):
        for code_idx in tqdm(range(512)):
            delta_embeddings = clip_embeddings[layer_idx, code_idx]

            projection = delta_embeddings.dot(text_direction)

            projections[layer_idx, code_idx] = projection

    min_projection = np.percentile(np.abs(projections), percentile)

    projections[np.abs(projections) < min_projection] = 0.0
    
    return projections

def plot_latent_directions(latent_strengths, output_dir, file_name):  
    plt.plot(np.arange(0, len(latent_strengths)), latent_strengths / np.max(latent_strengths))
    plt.savefig(os.path.join(output_dir, file_name))
    plt.clf()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='get clip boundary')

    parser.add_argument('--source_text', required=True, help='source clip text')
    parser.add_argument('--target_text', required=True, help='target clip text')

    parser.add_argument('--name', required=True, help="Path to output directory")

    parser.add_argument("--cutoff_percentile", type=int, default=80, help="Drop all boundary directions which are below this percentile in importance scores.")

    parser.add_argument("--precomputed_dirs", help="Optional path to precomputed feature directions matrix")

    parser.add_argument("--latent_adjustment_dir", help="Path to directory with modified latent embeddings (generated through clip). " \
                                                        "Will be used to compute feature directions")

    parser.add_argument("--out_dir", required=True, help="Path to directory where outputs will be placed")

    parser.add_argument("--model_layers", default=18, type=int, help="Number of W+ layers in the given model")

    args = parser.parse_args()

    if args.precomputed_dirs:
        feature_dir_array = np.load(args.precomputed_dirs)
    else:
        feature_dir_array = precompute_feature_directions(args.latent_adjustment_dir, args.model_layers)
        if args.feature_dir_out:
            np.save(os.path.join(args.out_dir, "feature_dirs.npy"), feature_dir_array)
    
    model, preprocess = clip.load("ViT-B/32", device='cuda:0')

    direction = get_textual_direction([args.target_text, args.source_text], imagenet_templates, model)

    boundary = get_boundary_for_direction_precomp(feature_dir_array, direction, args.model_layers, percentile=args.cutoff_percentile)
    
    boundary_dir = os.path.join(args.out_dir, args.name)

    os.makedirs(boundary_dir, exist_ok=True)

    np.save(os.path.join(boundary_dir, "boundary.npy"), boundary)
    np.save(os.path.join(boundary_dir, "intercept.npy"), boundary) # save a dummy intercept. Will not be used.

    plot_latent_directions(np.mean(boundary, axis=1), boundary_dir, "layer_strengths.png")