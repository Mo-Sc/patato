from typing import Sequence

import numpy as np

from ..reconstruction_algorithm import ReconstructionAlgorithm


class DeepMBReconstruction(ReconstructionAlgorithm):
    """
    DeepMB ONNX-based reconstruction algorithm

    Source:
    Dehner, C., Zahnd, G., Ntziachristos, V. et al.
    A deep neural network for real-time optoacoustic image reconstruction with adjustable speed of sound.
    Nat Mach Intell 5, 1130–1141 (2023).
    https://doi.org/10.1038/s42256-023-00724-3
    """

    N_PIXELS = (400, 1, 330)

    def __init__(
        self,
        n_pixels: Sequence[int],
        field_of_view: Sequence[float],
        speed_of_sound: float = 1540.0,
        *,
        model_path: str,
        use_gpu: bool = False,
        channels_to_interpolate: Sequence[int] = (),
        laser_energy: float = 1.0,
    ):
        """
        n_pixels: Native DeepMB output dimensions in (x, y, z) order.
        field_of_view: Reconstruction field of view in (x, y, z) order.
        speed_of_sound: Default speed of sound used by the reconstruction.
        model_path: Path to the ONNX model file.
        use_gpu: Whether to use GPU for inference (if available).
        channels_to_interpolate: Tuple of channel indices that should be interpolated by the model.
        laser_energy: Laser energy used during acquisition
        """

        if tuple(n_pixels) != self.N_PIXELS:
            raise ValueError(
                f"DeepMB produces {self.N_PIXELS} pixels in (x, y, z) order. Received {tuple(n_pixels)}"
            )
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise RuntimeError("DeepMB reconstruction requires onnxruntime") from e

        super().__init__(n_pixels, field_of_view, speed_of_sound)

        self.model_path = model_path
        self.laser_energy = float(laser_energy)

        # Providers
        if use_gpu:
            self.providers = [
                p
                for p in ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if p in ort.get_available_providers()
            ]
        else:
            self.providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(self.model_path, providers=self.providers)

        # Pre-build interpolation mask
        self.channels_for_interpolation = np.zeros(256, dtype=bool)
        self.channels_for_interpolation[list(channels_to_interpolate)] = True

        print("DeepMB running on:", self.session.get_providers()[0])

    def reconstruct(
        self,
        time_series: np.ndarray,
        fs: float = None,
        geometry: np.ndarray = None,
        n_pixels: Sequence[int] = None,
        field_of_view: Sequence[float] = None,
        speed_of_sound: float = None,
        **kwargs,
    ) -> np.ndarray:
        """DeepMB reconstruction entry point.

        Note: DeepMB is designed for a specific geometry.
        It will always assume that the input time series data corresponds to the geometry it was trained on
        and the output will always be (330, 400).
        time series can have any batch shape as long as the last two dimensions are (n_detectors, n_time_samples)
        """

        if n_pixels is not None or field_of_view is not None or geometry is not None:
            print(
                "Warning: n_pixels, field_of_view and geometry parameters are not used in DeepMB reconstruction. They are included for API consistency but will be ignored."
            )

        # PATimeSeries to numpy array if not already done
        if hasattr(time_series, "raw_data"):
            time_series = np.asarray(time_series.raw_data)

        if speed_of_sound is None:
            speed_of_sound = self.speed_of_sound

        # (..., n_detectors, n_time_samples)
        non_spatial_dims = time_series.shape[:-2]

        # reshape if necessary
        if len(non_spatial_dims) > 1:
            # (frame, wavelength, n_detectors, n_time_samples) -> (frame * wavelength, n_detectors, n_time_samples)
            frames = int(np.prod(non_spatial_dims))
            signal = time_series.reshape((frames,) + time_series.shape[-2:])
        elif len(non_spatial_dims) == 1:
            signal = time_series
        elif len(non_spatial_dims) == 0:
            signal = time_series[np.newaxis, ...]
        else:
            raise ValueError(
                f"Unsupported number of non-spatial dimensions: {len(non_spatial_dims)}"
            )

        outputs = []

        for sinogram in signal:

            inputs = {
                "sinogram": np.asarray(sinogram, dtype=np.float32),
                "speed_of_sound": np.asarray(speed_of_sound, dtype=np.float32),
                "laser_energy": np.asarray(self.laser_energy, dtype=np.float32),
                "channels_for_interpolation": self.channels_for_interpolation,
            }

            recon = self.session.run(None, inputs)[0]
            outputs.append(recon)

        # Stack frames and restore PATATO batch shape (frame, wavelength, x, y, z)
        output = np.stack(outputs, axis=0)
        output = output.reshape(non_spatial_dims + output.shape[1:])
        # add y axis for consistency
        output = np.expand_dims(output, axis=-2)
        # rotate by 180 degrees around y axis to match orientation of other reconstructions
        output = output[..., ::-1, :, ::-1]

        return output

    @staticmethod
    def get_algorithm_name() -> str:
        return "DeepMB ONNX Reconstruction"
