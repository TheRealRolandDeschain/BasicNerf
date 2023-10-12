#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# file: input_config.py
# Created on : 2023-10-11
# by : Johannes Zellinger
#
#
#
# --- imports -----------------------------------------------------------------
from typing import List

from pydantic import BaseModel


class EncoderConfigModel(BaseModel):
    input_dimensions: int
    number_of_frequencies: int
    frequencies_log_space: bool
    use_viewdirs: bool
    number_of_frequency_views: int


class StratifiedSamplingConfigModel(BaseModel):
    number_of_samples: int
    perturb: bool
    inverse_depth: bool


class ArchitectureConfigModel(BaseModel):
    filter_dimensions: int
    number_of_layers: int
    skip: List[int]
    use_fine_model: bool
    fine_filter_dimensions: int
    number_of_fine_layers: int


class HierarchicalSamplingConfigModel(BaseModel):
    number_of_hierarchical_samples: int
    perturb_hierarchical: bool


class OptimizerConfigModel(BaseModel):
    learning_rate: float


class TrainingConfigModel(BaseModel):
    number_of_samples: int
    test_image_idx: int
    near: float
    far: float
    number_of_iterations: int
    batch_size: int  # TODO validation to power of 2
    one_image_per_step: bool
    chunk_size: int
    center_crop: bool
    center_crop_iterations: int
    display_rate: int


class EarlyStoppingConfigModel(BaseModel):
    warmup_iterations: int
    warmup_min_fitness: float
    number_of_restarts: int


class ConfigModel(BaseModel):
    encoders: EncoderConfigModel
    stratified_sampling: StratifiedSamplingConfigModel
    model: ArchitectureConfigModel
    hierarchical_sampling: HierarchicalSamplingConfigModel
    optimizer: OptimizerConfigModel
    training: TrainingConfigModel
    early_stopping: EarlyStoppingConfigModel
