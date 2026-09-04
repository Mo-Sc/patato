#  Copyright (c) Thomas Else 2023-25.
#  License: MIT

"""
Translation of PATATO scan metadata into the IPASC consensus metadata dictionary.

This is the single source of truth for the mapping. It is used both to embed IPASC metadata
groups into PATATO's own HDF5 files and to build native IPASC exports, so the two cannot
drift apart. Only metadata that PATATO actually holds are emitted: IPASC's "report if
present" rule means an absent field must be omitted rather than guessed at.
"""

import uuid

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


# Identifiers that name the instrument itself. The transducer and laser serials are recorded
# too, but describe swappable components, and the software version changes without the
# hardware changing, so none of them takes part in the device identity.
_DEVICE_IDENTITY_KEYS = ("DeviceSN", "DAQ_MACaddress")


def _device_uuid(device_info: dict) -> str | None:
    """Map the identifiers a scanner recorded about itself onto an IPASC device UUID.

    IPASC wants the 8-4-4-4-12 form, which no manufacturer serial follows, so the recorded
    identifiers are hashed into a version 5 UUID. The result is deterministic, so every scan
    from one instrument carries the same device UUID, and it is derived from what the file
    actually records rather than invented. None when the source records nothing.
    """
    identity = {k: device_info[k] for k in _DEVICE_IDENTITY_KEYS if device_info.get(k)}
    if not identity:
        return None
    name = "|".join(f"{key}={identity[key]}" for key in sorted(identity))
    return str(uuid.uuid5(uuid.NAMESPACE_OID, name))


def _cpp_data_type(dtype) -> str:
    """Name a numpy dtype the way IPASC does, in C++ terms.

    An unmapped dtype raises rather than falling back to a guess: naming the wrong width
    would make the binary data unreadable to anything trusting the metadata.
    """
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

    timestamps = timestamps_to_ipasc(np.asarray(reader.get_scan_times())[:, 0])
    if timestamps is not None:
        acquisition[Tags.MEASUREMENT_TIMESTAMPS.tag] = timestamps

    speed_of_sound = reader.get_speed_of_sound()
    if speed_of_sound is not None:
        acquisition[Tags.SPEED_OF_SOUND.tag] = float(speed_of_sound)

    temperature = temperature_to_ipasc(np.asarray(reader.get_temperature())[:, 0])
    if temperature is not None:
        acquisition[Tags.TEMPERATURE_CONTROL.tag] = temperature

    # No IPASC tag describes a per-acquisition gain, so it travels as a custom parameter.
    # Without it a reconstruction from the exported file would apply no energy correction.
    correction_factor = np.asarray(reader.get_correction_factor(), dtype=float)
    if np.isfinite(correction_factor).all():
        acquisition[IPASCTags.CORRECTION_FACTOR] = correction_factor

    # Spatial poses are deliberately not reported. IPASC requires metres, but PATATO passes
    # the scanner z position through from the vendor without converting it (iThera reports
    # stage positions in millimetres), so the unit cannot be guaranteed. Emitting an
    # unverified unit would silently corrupt the geometry of anything reading the file.

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
        # Carried verbatim as a custom parameter as well, because the UUID is a one way
        # mapping and the serials are what a reader can actually check against a scanner.
        general[IPASCTags.DEVICE_INFO] = dict(device_info)
    device_uuid = _device_uuid(device_info)
    if device_uuid is not None:
        general[pf.MetadataDeviceTags.UNIQUE_IDENTIFIER.tag] = device_uuid

    return acquisition, device


def describe_ipasc_metadata(reader) -> list[dict]:
    """
    Describe every IPASC acquisition metadatum for a scan, present or not.

    Each entry carries the IPASC tag name, the value PATATO can supply, the SI unit and
    whether IPASC considers the field minimal. Absent fields are reported with a value of
    None rather than dropped, so a viewer can show what a dataset is missing.
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
