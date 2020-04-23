from .hourglass import HGNet
from .hourglass_gated import GatedHGNet


def get_model(name):
    if name == "HGNet":
        return HGNet
    elif name == "GatedHGNet":
        return GatedHGNet
    else:
        raise ValueError(f"{name} not found in available models")