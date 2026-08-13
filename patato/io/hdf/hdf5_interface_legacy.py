# Copyright (c) Thomas Else 2023-25.
# License: MIT

from __future__ import annotations

import warnings

import h5py
import numpy as np

from ...core.image_structures.image_structure_types import IMAGE_DATA_TYPES
from ...io.attribute_tags import HDF5Tags, ROITags
from ...io.attribute_tags_orig import (
    HDF5Tags as LegacyHDF5Tags,
    ROITags as LegacyROITags,
)
from ...io.hdf.fileimporter import ReaderInterface
from ...utils.rois.roi_type import ROI
from .hdf5_interface import load_image_from_hdf5


_LEGACY_IMAGE_GROUPS = {
    HDF5Tags.RECONSTRUCTION: LegacyHDF5Tags.RECONSTRUCTION,
    HDF5Tags.ULTRASOUND: LegacyHDF5Tags.ULTRASOUND,
}


class HDF5ReaderLegacy(ReaderInterface):
    def __init__(self, file):
        super().__init__()
        if isinstance(file, h5py.File):
            self.file = file
            self._own_file = False
        else:
            self.file = h5py.File(file, "r")
            self._own_file = True

    @staticmethod
    def _roi_frames(match_frames, frame_type, value, n_frames):
        if match_frames is None:
            return np.arange(n_frames, dtype=int)

        frame_values = np.asarray(match_frames)[:, 0]
        try:
            matches = np.flatnonzero(np.isclose(frame_values, value))
        except (TypeError, ValueError):
            matches = np.asarray([], dtype=int)
        if matches.size == 1:
            return matches.astype(int)

        warnings.warn(
            f"Could not uniquely match legacy ROI using {frame_type}; "
            "assigning it to all frames.",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.arange(n_frames, dtype=int)

    def _get_rois(self):
        output = {}
        if LegacyHDF5Tags.REGIONS_OF_INTEREST not in self.file:
            return output

        try:
            clinical = self.is_clinical()
            frame_type = (
                LegacyROITags.REPETITION
                if clinical
                else LegacyROITags.Z_POSITION
            )
            match_frames = (
                self.get_repetition_numbers()
                if clinical
                else self.get_scanner_z_position()
            )
            n_frames = match_frames.shape[0]
        except Exception:
            frame_type = None
            match_frames = None
            n_frames = self._get_pa_data()[0].shape[0]

        roi_root = self.file[LegacyHDF5Tags.REGIONS_OF_INTEREST]
        for roi_name in roi_root:
            roi_group = roi_root[roi_name]
            for roi_number in roi_group:
                dataset = roi_group[roi_number]
                legacy_value = (
                    dataset.attrs.get(frame_type, 1.0)
                    if frame_type is not None
                    else 1.0
                )
                ax0_indices = self._roi_frames(
                    match_frames, frame_type, legacy_value, n_frames
                )
                output[(roi_name, roi_number)] = ROI(
                    dataset[:],
                    dataset.attrs.get(LegacyROITags.Z_POSITION, np.nan),
                    dataset.attrs.get(LegacyROITags.RUN, np.nan),
                    dataset.attrs.get(LegacyROITags.REPETITION, np.nan),
                    dataset.attrs.get(LegacyROITags.ROI_NAME, "Unknown"),
                    dataset.attrs.get(LegacyROITags.ROI_POSITION, "Unknown"),
                    dataset.attrs.get(LegacyROITags.GENERATED_ROI, False),
                    ax0_indices,
                    dataset.attrs.get(ROITags.ROI_TYPE, "Unknown"),
                    dataset.attrs.get(ROITags.ROI_GROUP),
                    dataset.attrs.get(ROITags.ROI_ID),
                )
        return output

    def get_scan_datetime(self):
        import dateutil.parser

        try:
            value = self.file.attrs[LegacyHDF5Tags.DATE]
            if isinstance(value, str):
                return dateutil.parser.isoparse(value).replace(tzinfo=None)
            return value
        except KeyError:
            return np.nan

    def _get_pa_data(self):
        dataset = self.file[LegacyHDF5Tags.RAW_DATA]
        return dataset, dict(dataset.attrs)

    def get_scan_name(self):
        return self.file[LegacyHDF5Tags.RAW_DATA].attrs[LegacyHDF5Tags.SCAN_NAME]

    def _get_temperature(self):
        return self.file[LegacyHDF5Tags.TEMPERATURE]

    def _get_correction_factor(self):
        correction = self.file.get(LegacyHDF5Tags.OVERALL_CORR)
        if correction is None or np.any(np.isnan(correction[:])):
            return self.file[LegacyHDF5Tags.POWER][:]
        return correction[:]

    def _get_scanner_z_position(self):
        return self.file[LegacyHDF5Tags.Z_POSITION]

    def _get_run_numbers(self):
        if LegacyHDF5Tags.RUN in self.file:
            return self.file[LegacyHDF5Tags.RUN]

        n_frames, n_wavelengths = self.file[LegacyHDF5Tags.TIMESTAMP].shape[:2]
        return np.zeros((n_frames, n_wavelengths))

    def _get_repetition_numbers(self):
        return self.file[LegacyHDF5Tags.REPETITION]

    def _get_scan_times(self):
        return self.file[LegacyHDF5Tags.TIMESTAMP][:, :] * 1e-7

    def _get_sensor_geometry(self):
        return self.file[LegacyHDF5Tags.SCAN_GEOMETRY][:]

    def get_impulse_response(self):
        return self.file[LegacyHDF5Tags.IMPULSE_RESPONSE]

    def _get_wavelengths(self):
        return self.file[LegacyHDF5Tags.WAVELENGTH][:]

    def _get_water_absorption(self):
        water = self.file[LegacyHDF5Tags.WATER_ABSORPTION_COEFF]
        return water, water.attrs[LegacyHDF5Tags.WATER_PATHLENGTH]

    def _get_datasets(self):
        output = {image_type: {} for image_type in IMAGE_DATA_TYPES}
        for image_type, dtype in IMAGE_DATA_TYPES.items():
            group_name = _LEGACY_IMAGE_GROUPS.get(image_type, image_type)
            if group_name not in self.file:
                continue
            image_group = self.file[group_name]
            for recon in image_group:
                for recon_num in image_group[recon]:
                    image = load_image_from_hdf5(
                        dtype, image_group[recon][recon_num], self.file
                    )
                    output[image_type][(recon, recon_num)] = image
        return output

    def get_scan_comment(self):
        return self.file.attrs.get(LegacyHDF5Tags.SCAN_COMMENT, "")

    def _get_sampling_frequency(self):
        return self.file[LegacyHDF5Tags.RAW_DATA].attrs[
            LegacyHDF5Tags.SAMPLING_FREQ
        ]

    def get_speed_of_sound(self):
        return self.file[LegacyHDF5Tags.RAW_DATA].attrs.get(
            LegacyHDF5Tags.SPEED_OF_SOUND, None
        )

    def _get_segmentation(self):
        return self.file.get(LegacyHDF5Tags.SEGMENTATION, None)

    def close(self):
        if self._own_file:
            self.file.close()
