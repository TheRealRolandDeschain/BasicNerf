#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# file: main.py
# Created on : 2023-10-07
# by : Johannes Zellinger
#
#
#
# --- imports -----------------------------------------------------------------
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch import nn

from basic_nerf import renderer, utils


def nerf_forward(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: float,
    far: float,
    encoding_fn: Callable[[torch.Tensor], torch.Tensor],
    coarse_model: nn.Module,
    kwargs_sample_stratified: Optional[dict] = None,
    n_samples_hierarchical: int = 0,
    kwargs_sample_hierarchical: Optional[dict] = None,
    fine_model=None,
    viewdirs_encoding_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    chunksize: int = 2**15,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    r"""
    Compute forward pass through model(s).
    """

    # Set no kwargs if none are given.
    if kwargs_sample_stratified is None:
        kwargs_sample_stratified = {}
    if kwargs_sample_hierarchical is None:
        kwargs_sample_hierarchical = {}

    # Sample query points along each ray.
    query_points, z_vals = utils.sample_stratified(
        rays_o, rays_d, near, far, **kwargs_sample_stratified
    )

    # Prepare batches.
    batches = utils.prepare_chunks(query_points, encoding_fn, chunksize=chunksize)
    if viewdirs_encoding_fn is not None:
        batches_viewdirs = utils.prepare_viewdirs_chunks(
            query_points, rays_d, viewdirs_encoding_fn, chunksize=chunksize
        )
    else:
        batches_viewdirs = [None] * len(batches)

    # Coarse model pass.
    # Split the encoded points into "chunks", run the model on all chunks, and
    # concatenate the results (to avoid out-of-memory issues).
    predictions = []
    for batch, batch_viewdirs in zip(batches, batches_viewdirs):
        predictions.append(coarse_model(batch, viewdirs=batch_viewdirs))
    raw = torch.cat(predictions, dim=0)
    raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]])

    # Perform differentiable volume rendering to re-synthesize the RGB image.
    rgb_map, depth_map, acc_map, weights = renderer.raw2outputs(raw, z_vals, rays_d)
    # rgb_map, depth_map, acc_map, weights = render_volume_density(raw, rays_o, z_vals)
    outputs = {"z_vals_stratified": z_vals}

    # Fine model pass.
    if n_samples_hierarchical > 0:
        # Save previous outputs to return.
        rgb_map_0, depth_map_0, acc_map_0 = rgb_map, depth_map, acc_map

        # Apply hierarchical sampling for fine query points.
        query_points, z_vals_combined, z_hierarch = utils.sample_hierarchical(
            rays_o,
            rays_d,
            z_vals,
            weights,
            n_samples_hierarchical,
            **kwargs_sample_hierarchical,
        )

        # Prepare inputs as before.
        batches = utils.prepare_chunks(query_points, encoding_fn, chunksize=chunksize)
        if viewdirs_encoding_fn is not None:
            batches_viewdirs = utils.prepare_viewdirs_chunks(
                query_points, rays_d, viewdirs_encoding_fn, chunksize=chunksize
            )
        else:
            batches_viewdirs = [None] * len(batches)

        # Forward pass new samples through fine model.
        fine_model = fine_model if fine_model is not None else coarse_model
        predictions = []
        for batch, batch_viewdirs in zip(batches, batches_viewdirs):
            predictions.append(fine_model(batch, viewdirs=batch_viewdirs))
        raw = torch.cat(predictions, dim=0)
        raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]])

        # Perform differentiable volume rendering to re-synthesize the RGB image.
        rgb_map, depth_map, acc_map, weights = renderer.raw2outputs(
            raw, z_vals_combined, rays_d
        )

        # Store outputs.
        outputs["z_vals_hierarchical"] = z_hierarch
        outputs["rgb_map_0"] = rgb_map_0
        outputs["depth_map_0"] = depth_map_0
        outputs["acc_map_0"] = acc_map_0

    # Store outputs.
    outputs["rgb_map"] = rgb_map
    outputs["depth_map"] = depth_map
    outputs["acc_map"] = acc_map
    outputs["weights"] = weights
    return outputs  # TODO check types


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = np.load("data/tiny_nerf_data.npz")
    images = data["images"]
    poses = data["poses"]
    focal = data["focal"]

    print(f"Images shape: {images.shape}")
    print(f"Poses shape: {poses.shape}")
    print(f"Focal length: {focal}")

    height, width = images.shape[1:3]

    n_training = 100
    testimg_idx = 101

    # Gather as torch tensors
    images = torch.from_numpy(data["images"][:n_training]).to(device)
    poses = torch.from_numpy(data["poses"]).to(device)
    focal = torch.from_numpy(data["focal"]).to(device)
    torch.from_numpy(data["images"][testimg_idx]).to(device)
    testpose = torch.from_numpy(data["poses"][testimg_idx]).to(device)

    # Grab rays from sample image
    height, width = images.shape[1:3]
    with torch.no_grad():
        ray_origin, ray_direction = utils.get_rays(
            height, width, focal.item(), testpose
        )

    print("Ray Origin")
    print(ray_origin.shape)
    print(ray_origin[height // 2, width // 2, :])
    print("")

    print("Ray Direction")
    print(ray_direction.shape)
    print(ray_direction[height // 2, width // 2, :])
    print("")


if __name__ == "__main__":
    main()
