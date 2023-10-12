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

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import trange

from basic_nerf import EarlyStopping, NeRF, PositionalEncoder, renderer, utils
from basic_nerf.models import ConfigModel


def init_models(config: ConfigModel, device: torch.device):
    r"""
    Initialize models, encoders, and optimizer for NeRF training.
    """
    # Encoders
    encoder = PositionalEncoder(
        config.encoders.input_dimensions,
        config.encoders.number_of_frequencies,
        log_space=config.encoders.frequencies_log_space,
    )

    def encode(x):
        return encoder(x)

    # View direction encoders
    if config.encoders.use_viewdirs:
        encoder_viewdirs = PositionalEncoder(
            config.encoders.input_dimensions,
            config.encoders.number_of_frequency_views,
            log_space=config.encoders.frequencies_log_space,
        )

        def encode_viewdirs(x):
            return encoder_viewdirs(x)

        d_viewdirs = encoder_viewdirs.d_output
    else:
        encode_viewdirs = None
        d_viewdirs = None

    # Models
    model = NeRF(
        encoder.d_output,
        n_layers=config.model.number_of_layers,
        d_filter=config.model.filter_dimensions,
        skip=config.model.skip,
        d_viewdirs=d_viewdirs,
    )
    model.to(device)
    model_params = list(model.parameters())
    if config.model.use_fine_model:
        fine_model = NeRF(
            encoder.d_output,
            n_layers=config.model.number_of_layers,
            d_filter=config.model.fine_filter_dimensions,
            skip=config.model.skip,
            d_viewdirs=d_viewdirs,
        )
        fine_model.to(device)
        model_params = model_params + list(fine_model.parameters())
    else:
        fine_model = None

    # Optimizer
    optimizer = torch.optim.Adam(model_params, lr=config.optimizer.learning_rate)

    # Early Stopping
    warmup_stopper = EarlyStopping(patience=50)

    return model, fine_model, encode, encode_viewdirs, optimizer, warmup_stopper


def train(
    config: ConfigModel,
    device: torch.device,
    images: torch.Tensor,
    poses: torch.Tensor,
    test_image: torch.Tensor,
    test_pose: torch.Tensor,
    focal: torch.Tensor,
):
    r"""
    Launch training session for NeRF.
    """
    (
        model,
        fine_model,
        encode,
        encode_viewdirs,
        optimizer,
        warmup_stopper,
    ) = init_models(config, device)

    # Shuffle rays across all images.
    if not config.training.one_image_per_step:
        height, width = images.shape[1:3]
        all_rays = torch.stack(
            [
                torch.stack(utils.get_rays(height, width, focal, p), 0)
                for p in poses[: config.training.number_of_samples]
            ],
            0,
        )
        rays_rgb = torch.cat([all_rays, images[:, None]], 1)
        rays_rgb = torch.permute(rays_rgb, [0, 2, 3, 1, 4])
        rays_rgb = rays_rgb.reshape([-1, 3, 3])
        rays_rgb = rays_rgb.type(torch.float32)
        rays_rgb = rays_rgb[torch.randperm(rays_rgb.shape[0])]
        i_batch = 0

    train_psnrs = []
    val_psnrs = []
    iternums = []
    for i in trange(config.training.number_of_iterations):
        model.train()

        if config.training.one_image_per_step:
            # Randomly pick an image as the target.
            target_img_idx = np.random.randint(images.shape[0])
            target_img = images[target_img_idx].to(device)
            if (
                config.training.center_crop
                and i < config.training.center_crop_iterations
            ):
                target_img = utils.crop_center(target_img)
            height, width = target_img.shape[:2]
            target_pose = poses[target_img_idx].to(device)
            rays_o, rays_d = utils.get_rays(height, width, focal, target_pose)
            rays_o = rays_o.reshape([-1, 3])
            rays_d = rays_d.reshape([-1, 3])
        else:
            # Random over all images.
            batch = rays_rgb[i_batch : i_batch + config.training.batch_size]
            batch = torch.transpose(batch, 0, 1)
            rays_o, rays_d, target_img = batch
            height, width = target_img.shape[:2]
            i_batch += config.training.batch_size
            # Shuffle after one epoch
            if i_batch >= rays_rgb.shape[0]:
                rays_rgb = rays_rgb[torch.randperm(rays_rgb.shape[0])]
                i_batch = 0
        target_img = target_img.reshape([-1, 3])

        # Run one iteration of TinyNeRF and get the rendered RGB image.
        kwargs_sample_stratified = {
            "n_samples": config.stratified_sampling.number_of_samples,
            "perturb": config.stratified_sampling.perturb,
            "inverse_depth": config.stratified_sampling.inverse_depth,
        }
        kwargs_sample_hierarchical = {"perturb": config.stratified_sampling.perturb}
        outputs = nerf_forward(
            rays_o,
            rays_d,
            config.training.near,
            config.training.far,
            encode,
            model,
            kwargs_sample_stratified=kwargs_sample_stratified,
            n_samples_hierarchical=config.hierarchical_sampling.number_of_hierarchical_samples,
            kwargs_sample_hierarchical=kwargs_sample_hierarchical,
            fine_model=fine_model,
            viewdirs_encoding_fn=encode_viewdirs,
            chunksize=config.training.chunk_size,
        )

        # Check for any numerical issues.
        for k, v in outputs.items():
            if torch.isnan(v).any():
                print(f"! [Numerical Alert] {k} contains NaN.")
            if torch.isinf(v).any():
                print(f"! [Numerical Alert] {k} contains Inf.")

        # Backprop!
        rgb_predicted = outputs["rgb_map"]
        loss = torch.nn.functional.mse_loss(rgb_predicted, target_img)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        psnr = -10.0 * torch.log10(loss)
        train_psnrs.append(psnr.item())

        # Evaluate testimg at given display rate.
        if i % config.training.display_rate == 0:
            model.eval()
            height, width = test_image.shape[:2]
            rays_o, rays_d = utils.get_rays(height, width, focal, test_pose)
            rays_o = rays_o.reshape([-1, 3])
            rays_d = rays_d.reshape([-1, 3])
            outputs = nerf_forward(
                rays_o,
                rays_d,
                config.training.near,
                config.training.far,
                encode,
                model,
                kwargs_sample_stratified=kwargs_sample_stratified,
                n_samples_hierarchical=config.hierarchical_sampling.number_of_hierarchical_samples,
                kwargs_sample_hierarchical=kwargs_sample_hierarchical,
                fine_model=fine_model,
                viewdirs_encoding_fn=encode_viewdirs,
                chunksize=config.training.chunk_size,
            )

            rgb_predicted = outputs["rgb_map"]
            loss = torch.nn.functional.mse_loss(
                rgb_predicted, test_image.reshape(-1, 3)
            )
            print("Loss:", loss.item())
            val_psnr = -10.0 * torch.log10(loss)
            val_psnrs.append(val_psnr.item())
            iternums.append(i)

            # Plot example outputs
            fig, ax = plt.subplots(
                1, 4, figsize=(24, 4), gridspec_kw={"width_ratios": [1, 1, 1, 3]}
            )
            ax[0].imshow(
                rgb_predicted.reshape([height, width, 3]).detach().cpu().numpy()
            )
            ax[0].set_title(f"Iteration: {i}")
            ax[1].imshow(test_image.detach().cpu().numpy())
            ax[1].set_title("Target")
            ax[2].plot(range(0, i + 1), train_psnrs, "r")
            ax[2].plot(iternums, val_psnrs, "b")
            ax[2].set_title("PSNR (train=red, val=blue")
            z_vals_strat = outputs["z_vals_stratified"].view(
                (-1, config.stratified_sampling.number_of_samples)
            )
            z_sample_strat = (
                z_vals_strat[z_vals_strat.shape[0] // 2].detach().cpu().numpy()
            )
            if "z_vals_hierarchical" in outputs:
                z_vals_hierarch = outputs["z_vals_hierarchical"].view(
                    (-1, config.hierarchical_sampling.number_of_hierarchical_samples)
                )
                z_sample_hierarch = (
                    z_vals_hierarch[z_vals_hierarch.shape[0] // 2]
                    .detach()
                    .cpu()
                    .numpy()
                )
            else:
                z_sample_hierarch = None
            _ = utils.plot_samples(z_sample_strat, z_sample_hierarch, ax=ax[3])
            ax[3].margins(0)
            plt.savefig(f"results/epoch_{i}.png")

        # Check PSNR for issues and stop if any are found.
        if i == config.early_stopping.warmup_iterations - 1:
            if val_psnr < config.early_stopping.warmup_min_fitness:
                print(
                    f"Val PSNR {val_psnr} below warmup_min_fitness "
                    f"{config.early_stopping.warmup_min_fitness}. Stopping..."
                )
                return False, train_psnrs, val_psnrs
        elif i < config.early_stopping.warmup_iterations:
            if warmup_stopper is not None and warmup_stopper(i, psnr):
                print(
                    f"Train PSNR flatlined at {psnr} for "
                    f"{warmup_stopper.patience} iters. Stopping..."
                )
                return False, train_psnrs, val_psnrs

    return True, train_psnrs, val_psnrs


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
    print(device)
    with open(
        "config/default.json", "r"
    ) as input_config_file:  # TODO choose this file via cmd arg
        config = ConfigModel.model_validate_json(input_config_file.read())
    data = np.load("data/tiny_nerf_data.npz")
    images = data["images"]
    poses = data["poses"]
    focal = data["focal"]

    print(f"Images shape: {images.shape}")
    print(f"Poses shape: {poses.shape}")
    print(f"Focal length: {focal}")

    # Gather as torch tensors
    images = torch.from_numpy(data["images"][: config.training.number_of_samples]).to(
        device
    )
    poses = torch.from_numpy(data["poses"]).to(device)
    focal = torch.from_numpy(data["focal"]).to(device)
    test_image = torch.from_numpy(data["images"][config.training.test_image_idx]).to(
        device
    )
    test_pose = torch.from_numpy(data["poses"][config.training.test_image_idx]).to(
        device
    )

    # Run training session(s)
    for _ in range(config.early_stopping.number_of_restarts):
        success, train_psnrs, val_psnrs = train(
            config, device, images, poses, test_image, test_pose, focal
        )
        if success and val_psnrs[-1] >= config.early_stopping.warmup_min_fitness:
            print("Training successful!")
            print(train_psnrs)
            print(val_psnrs)
            break

    print("")
    print("Done!")


if __name__ == "__main__":
    main()
