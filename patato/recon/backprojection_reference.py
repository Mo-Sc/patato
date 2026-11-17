#  Copyright (c) Thomas Else 2023-25.
#  License: MIT

from typing import Sequence

import numpy as np

from .backprojection_implementation.jax_implementation import full_recon
from .reconstruction_algorithm import ReconstructionAlgorithm

try:
    import jax
except ImportError:
    print(
        """WARNING: JAX must be installed to support the standard backprojection algorithm and filtering.
    Alternatively, try the Numpy implementations.
    """
    )
    jax = None


class ReferenceBackprojection(ReconstructionAlgorithm):
    """Reference backprojection: Uses JAX in the background."""

    def reconstruct(
        self,
        time_series: np.ndarray,
        fs: float,
        geometry: np.ndarray,
        n_pixels: Sequence[int],
        field_of_view: Sequence[float],
        speed_of_sound,
        **kwargs,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        time_series
        fs
        geometry
        n_pixels
        field_of_view
        speed_of_sound
        kwargs.

        Returns
        -------
        """

        # Get parameters:
        dl = speed_of_sound / fs

        # Reshape frames so that we can loop through to reconstruct
        original_shape = time_series.shape[:-2]
        frames = int(np.prod(original_shape))
        signal = time_series.reshape((frames,) + time_series.shape[-2:])
        # The field of view spans the outer pixel edges, as in iThera's own images:
        # spacing fov / n, pixel centres half a pixel inside (full_recon centres the grid).
        dx, dy, dz = (f / n for f, n in zip(field_of_view, n_pixels))

        recon_all = jax.vmap(full_recon, in_axes=(0,) + (None,) * 8, out_axes=0)

        output = recon_all(
            signal, geometry, dl, n_pixels[0], n_pixels[1], n_pixels[2], dx, dy, dz
        )

        return output.reshape(original_shape + tuple(n_pixels)[::-1])

    @staticmethod
    def get_algorithm_name() -> str:
        """
        Returns
        -------
        """
        return "Reference Backprojection"
