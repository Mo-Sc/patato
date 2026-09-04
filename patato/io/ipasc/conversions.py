#  Copyright (c) Thomas Else 2023-25.
#  License: MIT

"""
Unit conversions between PATATO's internal conventions and the IPASC metadata standard.

IPASC requires SI units throughout (metres, seconds since the Unix epoch, kelvin). PATATO
follows the acquisition conventions of the vendor formats it reads: wavelengths in
nanometres, acquisition times in seconds on the .NET epoch (the raw iThera tick counters
divided by 1e7), and temperatures in degrees Celsius. Every IPASC read or write must pass
through this module so the two directions cannot drift apart.
"""

import numpy as np

# Seconds between 0001-01-01 and 1970-01-01. PATATO timestamps are on the former epoch.
POSIX_EPOCH_OFFSET_S = 62135596800.0

ABSOLUTE_ZERO_C = -273.15


def wavelengths_to_ipasc(wavelengths_nm) -> np.ndarray:
    return np.asarray(wavelengths_nm, dtype=float) * 1e-9


def wavelengths_from_ipasc(wavelengths_m) -> np.ndarray:
    return np.asarray(wavelengths_m, dtype=float) * 1e9


def timestamps_to_ipasc(times_s):
    """
    Convert acquisition times to seconds since the Unix epoch, or None if they are relative.

    Vendor readers are not required to put acquisition times on an absolute epoch; some
    legacy files count from the start of the scan, and a reader with no times at all reports
    NaN. Shifting the former by the epoch offset would claim an acquisition date before 1970,
    and writing the latter would record a measurement that was never made, so neither is
    reported.
    """
    converted = np.asarray(times_s, dtype=float) - POSIX_EPOCH_OFFSET_S
    if not np.all(np.isfinite(converted)) or np.any(converted < 0):
        return None
    return converted


def timestamps_from_ipasc(times_s) -> np.ndarray:
    return np.asarray(times_s, dtype=float) + POSIX_EPOCH_OFFSET_S


def temperature_to_ipasc(temperature_c):
    """
    Convert acquisition temperatures to kelvin, or return None if they were never recorded.

    Vendor files routinely carry an all-zero or all-NaN temperature array when the probe has
    no temperature sensor. Reporting that as 273.15 K would assert a measurement that was
    never made, so an unpopulated array is omitted from the metadata instead.
    """
    temperature_c = np.asarray(temperature_c, dtype=float)
    if temperature_c.size == 0 or np.all(np.isnan(temperature_c) | (temperature_c == 0)):
        return None
    return temperature_c - ABSOLUTE_ZERO_C


def temperature_from_ipasc(temperature_k) -> np.ndarray:
    return np.asarray(temperature_k, dtype=float) + ABSOLUTE_ZERO_C
