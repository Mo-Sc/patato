from typing import Sequence
import numpy as np
from patato.recon import ReconstructionAlgorithm


class DeepMBReconstruction(ReconstructionAlgorithm):
    """
    DeepMB ONNX-based reconstruction algorithm

    Source:
    Dehner, C., Zahnd, G., Ntziachristos, V. et al.
    A deep neural network for real-time optoacoustic image reconstruction with adjustable speed of sound.
    Nat Mach Intell 5, 1130–1141 (2023).
    https://doi.org/10.1038/s42256-023-00724-3
    """

    def __init__(
        self,
        model_path: str,
        use_gpu: bool = False,
        channels_to_interpolate=(60, 148),
        laser_energy: float = 1.0,
    ):

        try:
            import onnxruntime as ort
        except ImportError as e:
            raise RuntimeError("DeepMB reconstruction requires onnxruntime") from e

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
        fs: float,
        geometry: np.ndarray,
        n_pixels: Sequence[int],
        field_of_view: Sequence[float],
        speed_of_sound: float,
        **kwargs,
    ) -> np.ndarray:
        """
        PATATO reconstruction entry point.
        """

        # PATATO shape: (..., n_detectors, n_time_samples)
        original_shape = time_series.shape[:-2]
        frames = int(np.prod(original_shape))

        signal = time_series.reshape((frames,) + time_series.shape[-2:])

        outputs = []

        for i in range(frames):
            # DeepMB expects (256, 2030)
            sinogram = signal[i].T.astype(np.float32)

            inputs = {
                "sinogram": sinogram,
                "speed_of_sound": np.array(speed_of_sound, dtype=np.float32),
                "laser_energy": np.array(self.laser_energy, dtype=np.float32),
                "channels_for_interpolation": self.channels_for_interpolation,
            }

            recon = self.session.run(None, inputs)[0]  # (330, 400)
            outputs.append(recon)

        # Stack frames and restore PATATO batch shape
        output = np.stack(outputs, axis=0)
        return output.reshape(original_shape + output.shape[1:])

    @staticmethod
    def get_algorithm_name() -> str:
        return "DeepMB ONNX Reconstruction"
