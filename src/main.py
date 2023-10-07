#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# file: main.py
# Created on : 2023-10-07
# by : Johannes Zellinger
#
#
#
# --- imports -----------------------------------------------------------------

import torch


def main():
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(torch_device)
    torch.device(torch_device)


if __name__ == "__main__":
    main()
