import torch
from torch import nn


def get_module_device(module: nn.Module) -> torch.device:
    """Get the device of a module.

    Args:
        module (nn.Module): A module contains the parameters.

    Returns:
        torch.device: The device of the module.
    """
    try:
        return next(module.parameters()).device
    except StopIteration:
        raise ValueError("The input module should contain parameters.") from None
