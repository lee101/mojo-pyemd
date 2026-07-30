"""Earth Mover's Distance with a Mojo compute core."""

from .emd import emd, emd_samples, emd_with_flow

__all__ = ["emd", "emd_with_flow", "emd_samples"]
__version__ = "0.1.0"
