"""Utility functions for model analysis."""

import torch


def param_matrix(model):
    """Print parameter matrix.
    
    Args:
        model: neural network built with pytorch.
    """
    for i in range(len(list(model.parameters()))):
        print(list(model.parameters())[i].size())


def total_num_param(model):
    """Print total number of parameters.
    
    Args:
        model: neural network built with pytorch.
    """
    print(sum(map(torch.numel, model.parameters())))


def param_trainable(model):
    """Print all trainable parameters and layer information.
    Args:
        model: neural network built with pytorch.
    """
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
            print(param.data)
            print(param.size())
            print("===============================================")
