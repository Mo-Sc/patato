from __future__ import annotations

import json
import logging

import h5py

from ...io.attribute_tags import HDF5Tags, IPASCTags
from ...io.attribute_tags_orig import HDF5Tags as LegacyHDF5Tags
from .hdf5_interface import HDF5Reader
from .hdf5_interface_legacy import HDF5ReaderLegacy
from ..ipasc.read_ipasc import IPASCInterface


logger = logging.getLogger(__name__)

_LEGACY_SIGNATURES = {
    LegacyHDF5Tags.OVERALL_CORR,
    LegacyHDF5Tags.TEMPERATURE,
    LegacyHDF5Tags.Z_POSITION,
    LegacyHDF5Tags.RUN,
    LegacyHDF5Tags.REPETITION,
    LegacyHDF5Tags.SCAN_GEOMETRY,
    LegacyHDF5Tags.IMPULSE_RESPONSE,
    LegacyHDF5Tags.POWER,
    LegacyHDF5Tags.RECONSTRUCTION,
    LegacyHDF5Tags.ULTRASOUND,
}


def is_ipasc_hdf5(file) -> bool:
    """
    Check if a given HDF5 file follows the IPASC consensus format.

    The IPASC format stores the raw time series under a fixed dataset name
    alongside an acquisition metadata group, neither of which PATATO uses.
    """
    return IPASCTags.BINARY_DATA in file and IPASCTags.META_DATA in file


def is_legacy_hdf5(file) -> bool:
    """
    Check if a given HDF5 file is a legacy PATATO scan by looking for origin_file.
    """
    value = file.attrs.get(HDF5Tags.FILE_ORIGIN)
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    try:
        origin = json.loads(value) if isinstance(value, str) else None
    except json.JSONDecodeError:
        origin = None
    if origin is not None and origin.get("tool") == "PATARI":
        return False

    # fallback to detect PATARI scans before the origin attribute was added
    # TODO: remove
    raw_data = file.get(HDF5Tags.RAW_DATA)
    if raw_data is not None and LegacyHDF5Tags.SCAN_NAME in raw_data.attrs:
        return True
    return any(signature in file for signature in _LEGACY_SIGNATURES)


def get_hdf5_reader(file, mode="r"):
    """
    factory function to get the appropriate HDF5 reader for a given file.
    Only necessary for compatibility with original PATATO scans
    Maybe remove this in the future
    """
    owns_file = not isinstance(file, h5py.File)
    if owns_file:
        file = h5py.File(file, mode)

    if is_ipasc_hdf5(file):
        # pacfish opens the file itself, so hand over the path rather than the handle.
        filename = file.filename
        if owns_file:
            file.close()
        return IPASCInterface(filename)

    if is_legacy_hdf5(file):
        logger.warning(
            "Legacy PATATO scan loaded; not all features might be available."
        )
        reader_type = HDF5ReaderLegacy
    else:
        reader_type = HDF5Reader

    reader = reader_type(file)
    reader._own_file = owns_file
    return reader
