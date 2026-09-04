#  Copyright (c) Thomas Else 2023-25.
#  Copyright (c) Janek Grohl 2023.
#  License: MIT

import logging
from os.path import split

import numpy as np

from ..attribute_tags import IPASCTags
from ..hdf.fileimporter import ReaderInterface
from .conversions import (
    temperature_from_ipasc,
    timestamps_from_ipasc,
    wavelengths_from_ipasc,
)

logger = logging.getLogger(__name__)


class IPASCInterface(ReaderInterface):
    """An interface for datasets in the IPASC consensus format.

    The format covers raw time series data only, so reconstructions, ultrasound,
    segmentations and annotations are absent by construction. Metadata are converted from
    IPASC's SI units into PATATO's internal conventions on the way in.
    """

    def __init__(self, file_path):
        super().__init__()
        import pacfish as pf

        self.scan_name = split(file_path)[-1]
        self.pa_data = pf.load_data(file_path)
        # IPASC binary data are [detectors, samples, wavelengths, frames]; PATATO indexes
        # [frames, wavelengths, detectors, samples]. pacfish reads the whole array into
        # memory anyway, so transpose it once into a contiguous block: reading it back as a
        # strided view costs about four times as much on every access.
        _, self.nsamples, self.nwavelengths, self.nframes = np.shape(
            self.pa_data.binary_time_series_data
        )
        self._time_series = np.ascontiguousarray(
            np.transpose(self.pa_data.binary_time_series_data, (3, 2, 0, 1))
        )
        self.pa_data.binary_time_series_data = None
        self.geometry = self.pa_data.get_detector_position()

        if (
            self.pa_data.get_custom_meta_datum(IPASCTags.CORRECTION_FACTOR) is None
            and self.pa_data.get_overall_gain() is None
        ):
            logger.warning(
                "%s records no gain; reconstructions will carry no energy correction",
                self.scan_name,
            )

    def is_clinical(self):
        # The base class infers this from the scanner positions, which an IPASC file does
        # not let us read. It does record what the frame axis means, so use that directly.
        return self.pa_data.get_dimensionality() == "time"

    def _per_acquisition(self, values):
        """Expand a per-frame metadatum onto PATATO's (frames, wavelengths) grid.

        IPASC treats wavelength and frame as independent axes, whereas PATATO indexes every
        acquisition by the pair, so a scalar or per-frame array is repeated across
        wavelengths.
        """
        shape = (self.nframes, self.nwavelengths)
        if values is None:
            return np.full(shape, np.nan)
        values = np.asarray(values, dtype=float)
        if values.size == 1:
            return np.full(shape, values.reshape(-1)[0])
        if values.size == self.nframes:
            return np.repeat(values.reshape(-1, 1), self.nwavelengths, axis=1)
        return values.reshape(shape)

    def get_n_samples(self):
        return self.nsamples

    def _get_pa_data(self):
        return self._time_series, {"fs": self._get_sampling_frequency()}

    def _get_wavelengths(self):
        return wavelengths_from_ipasc(
            np.atleast_1d(self.pa_data.get_acquisition_wavelengths())
        )

    def _get_sampling_frequency(self):
        return self.pa_data.get_sampling_rate()

    def _get_sensor_geometry(self):
        return np.asarray(self.geometry)

    def get_speed_of_sound(self):
        return self.pa_data.get_speed_of_sound()

    def _get_scan_times(self):
        return timestamps_from_ipasc(
            self._per_acquisition(self.pa_data.get_measurement_time_stamps())
        )

    def _get_temperature(self):
        return temperature_from_ipasc(
            self._per_acquisition(self.pa_data.get_temperature())
        )

    def _get_correction_factor(self):
        factor = self.pa_data.get_custom_meta_datum(IPASCTags.CORRECTION_FACTOR)
        if factor is not None:
            return np.asarray(factor, dtype=float).reshape(
                self.nframes, self.nwavelengths
            )
        gain = self.pa_data.get_overall_gain()
        if gain is not None:
            return self._per_acquisition(gain)
        # Ones leave the time series untouched. NaN would propagate through the energy
        # correction into every reconstructed pixel.
        return np.ones((self.nframes, self.nwavelengths))

    def _get_scanner_z_position(self):
        # IPASC records a 6D pose per frame but does not define the order of its components,
        # so which one is the scanner axis cannot be established. NaN marks it unknown, which
        # is also what `is_clinical` expects for a probe that does not translate.
        return np.full((self.nframes, self.nwavelengths), np.nan)

    def _get_run_numbers(self):
        # IPASC has no notion of runs or repetitions. Zeros mark them unknown, matching what
        # the HDF5 reader falls back to for files that never stored them.
        return np.zeros((self.nframes, self.nwavelengths), dtype=int)

    def _get_repetition_numbers(self):
        return np.zeros((self.nframes, self.nwavelengths), dtype=int)

    def get_impulse_response(self):
        # IPASC defines a detector frequency response, which is not the time domain impulse
        # response that PATATO deconvolves with.
        return None

    def _get_water_absorption(self):
        return None, None

    def _get_rois(self):
        # IPASC regions of interest are named cuboids in device coordinates, not the image
        # space polygons that PATATO annotates with.
        return {}

    def _get_segmentation(self):
        return None

    def _get_datasets(self):
        return {}

    def get_scan_name(self):
        return self.scan_name

    def get_scan_datetime(self):
        from datetime import datetime, timezone

        timestamps = self.pa_data.get_measurement_time_stamps()
        if timestamps is None:
            return np.nan
        first = float(np.asarray(timestamps).reshape(-1)[0])
        return datetime.fromtimestamp(first, timezone.utc).replace(tzinfo=None)

    def get_scan_comment(self):
        return ""

    def close(self):
        pass
