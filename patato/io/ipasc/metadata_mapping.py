"""
Translation of PATATO scan metadata into the IPASC metadata.
"""

import uuid
from datetime import datetime

import numpy as np

from ..attribute_tags import IPASCTags
from .conversions import (
    temperature_to_ipasc,
    timestamps_to_ipasc,
    wavelengths_to_ipasc,
)

# IPASC gives the data type in C++ naming.
_CPP_DATA_TYPES = {
    "int16": "short",
    "uint16": "unsigned short",
    "int32": "int",
    "uint32": "unsigned int",
    "int64": "long long",
    "uint64": "unsigned long long",
    "float32": "float",
    "float64": "double",
}


def _cpp_data_type(dtype) -> str:
    """Name a numpy dtype the way IPASC does, in C++ terms."""
    name = np.dtype(dtype).name
    if name not in _CPP_DATA_TYPES:
        raise ValueError(f"no IPASC data type name for '{name}'")
    return _CPP_DATA_TYPES[name]


def build_ipasc_metadata(reader) -> tuple[dict, dict]:
    """
    Build the IPASC acquisition and device metadata dictionaries for a PATATO reader.

    Returns
    -------
    (acquisition, device)
        Nested dictionaries keyed by IPASC tag names, with all values in SI units.
    """
    import pacfish as pf
    from pacfish import MetadataAcquisitionTags as Tags

    time_series = reader.get_pa_data()
    n_frames, n_wavelengths, n_detectors, n_samples = time_series.shape
    geometry = np.asarray(reader.get_sensor_geometry(), dtype=float)

    acquisition = {
        Tags.UUID.tag: str(uuid.uuid4()),
        Tags.ENCODING.tag: "UTF-8",
        Tags.COMPRESSION.tag: "raw",
        Tags.DATA_TYPE.tag: _cpp_data_type(time_series.dtype),
        # Derived from the recorded scanner positions: frames taken at one position differ
        # in time, frames taken at several differ in space. A reader that does not record
        # positions reports "time". IPASC's third option, "time and space", cannot be told
        # apart from "space" without knowing the acquisition order.
        Tags.DIMENSIONALITY.tag: "time" if reader.is_clinical() else "space",
        Tags.SIZES.tag: np.array([n_detectors, n_samples, n_wavelengths, n_frames]),
        Tags.AD_SAMPLING_RATE.tag: float(reader.get_sampling_frequency()),
        Tags.ACQUISITION_WAVELENGTHS.tag: wavelengths_to_ipasc(reader.get_wavelengths()),
    }

    # Scanners stamp frames with their local clock, so the acquisition datetime's offset is
    # what makes the times absolute. A reader that records no date cannot supply one.
    scan_datetime = reader.get_scan_datetime()
    utc_offset = (
        scan_datetime.utcoffset() if isinstance(scan_datetime, datetime) else None
    )
    timestamps = timestamps_to_ipasc(
        np.asarray(reader.get_scan_times())[:, 0], utc_offset
    )
    if timestamps is not None:
        acquisition[Tags.MEASUREMENT_TIMESTAMPS.tag] = timestamps

    speed_of_sound = reader.get_speed_of_sound()
    if speed_of_sound is not None:
        acquisition[Tags.SPEED_OF_SOUND.tag] = float(speed_of_sound)

    temperature = temperature_to_ipasc(np.asarray(reader.get_temperature())[:, 0])
    if temperature is not None:
        acquisition[Tags.TEMPERATURE_CONTROL.tag] = temperature

    # No IPASC tag describes a per-acquisition gain, so it becomes a custom parameter.
    # Without it a reconstruction from the exported file would apply no energy correction.
    correction_factor = np.asarray(reader.get_correction_factor(), dtype=float)
    if np.isfinite(correction_factor).all():
        acquisition[IPASCTags.CORRECTION_FACTOR] = correction_factor

    creator = pf.DeviceMetaDataCreator()
    for position in geometry:
        element = pf.DetectionElementCreator()
        element.set_detector_position(position)
        creator.add_detection_element(element.get_dictionary())
    device = creator.finalize_device_meta_data()

    # No field of view is reported. IPASC defines it as the volume the device can detect,
    # but every field of view a scan records (per reconstruction, or for the ultrasound
    # image) is a grid someone chose, not a property of the hardware.
    general = device[pf.MetadataDeviceTags.GENERAL.tag]
    device_info = reader.get_device_info()
    if device_info:
        # No IPASC tag holds vendor serials, so they travel as a custom parameter.
        general[IPASCTags.DEVICE_INFO] = dict(device_info)
    device_uuid = reader.get_device_uuid()
    if device_uuid is not None:
        general[pf.MetadataDeviceTags.UNIQUE_IDENTIFIER.tag] = device_uuid

    return acquisition, device


def describe_ipasc_metadata(reader) -> list[dict]:
    """
    Describe every IPASC acquisition metadatum for a scan, present or not.

    Each entry carries the IPASC tag name, the value PATATO can supply, the SI unit and
    whether IPASC considers the field minimal.
    """
    from pacfish import MetadataAcquisitionTags as Tags
    from pacfish import MetadataDeviceTags as DeviceTags

    acquisition, device = build_ipasc_metadata(reader)
    described = [
        {
            "tag": datum.tag,
            "value": acquisition.get(datum.tag),
            "unit": datum.unit,
            "minimal": datum.mandatory,
            "present": datum.tag in acquisition,
        }
        for datum in sorted(Tags.TAGS, key=lambda d: d.tag)
    ]
    general = device[DeviceTags.GENERAL.tag]
    described += [
        {
            "tag": f"device.{datum.tag}",
            "value": general.get(datum.tag),
            "unit": datum.unit,
            "minimal": datum.mandatory,
            "present": datum.tag in general,
        }
        for datum in sorted(
            (
                DeviceTags.UNIQUE_IDENTIFIER,
                DeviceTags.FIELD_OF_VIEW,
                DeviceTags.NUMBER_OF_DETECTION_ELEMENTS,
            ),
            key=lambda d: d.tag,
        )
    ]
    return described
