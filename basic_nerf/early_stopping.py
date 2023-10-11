#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# file: early_stopping.py
# Created on : 2023-10-07
# by : Johannes Zellinger
#
#
#
# --- imports -----------------------------------------------------------------


class EarlyStopping:
    def __init__(self, patience: int = 30, margin: float = 1e-4):
        self.best_fitness = 0.0  # In our case PSNR
        self.best_iter = 0
        self.margin = margin
        self.patience = patience or float(
            "inf"
        )  # epochs to wait after fitness stops improving to stop

    def __call__(self, iter: int, fitness: float):
        r"""
        Check if criterion for stopping is met.
        """
        if (fitness - self.best_fitness) > self.margin:
            self.best_iter = iter
            self.best_fitness = fitness
        delta = iter - self.best_iter
        stop = delta >= self.patience  # stop training if patience exceeded
        return stop
