"""Utility functions for model analysis."""


import torch


def param_matrix(model):
    """Print parameter matrix."""
    for i in range(len(list(model.parameters()))):
        print(list(model.parameters())[i].size())


def total_num_param(model):
    """Print total number of parameters."""
    print(sum(map(torch.numel, model.parameters())))


def param_trainable(model):
    """Print all trainable parameters and layer information."""
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
            print(param.data)
            print(param.size())
            print("===============================================")
