"""
Unit conversions between PATATO's internal conventions and the IPASC metadata standard.

IPASC requires SI units (metres, seconds since the Unix epoch in UTC, kelvin).
PATATO follows the acquisition conventions of the vendor formats it reads: wavelengths in
nanometres, acquisition times in seconds on the .NET epoch (the raw iThera tick counters
divided by 1e7) on the scanner's local wall clock, and temperatures in degrees Celsius.
Every IPASC read or write must pass through this module
"""

import numpy as np

# Seconds between 0001-01-01 and 1970-01-01. PATATO timestamps are on the former epoch.
POSIX_EPOCH_OFFSET_S = 62135596800.0

ABSOLUTE_ZERO_C = -273.15


def wavelengths_to_ipasc(wavelengths_nm) -> np.ndarray:
    return np.asarray(wavelengths_nm, dtype=float) * 1e-9


def wavelengths_from_ipasc(wavelengths_m) -> np.ndarray:
    return np.asarray(wavelengths_m, dtype=float) * 1e9


def timestamps_to_ipasc(times_s, utc_offset):
    """
    Convert local wall clock acquisition times to UTC seconds since the Unix epoch.

    Returns None when the result would not be a real timestamp
    """
    if utc_offset is None:
        return None
    converted = (
        np.asarray(times_s, dtype=float)
        - POSIX_EPOCH_OFFSET_S
        - utc_offset.total_seconds()
    )
    if not np.all(np.isfinite(converted)) or np.any(converted < 0):
        return None
    return converted


def timestamps_from_ipasc(times_s) -> np.ndarray:
    return np.asarray(times_s, dtype=float) + POSIX_EPOCH_OFFSET_S


def temperature_to_ipasc(temperature_c):
    """
    Convert acquisition temperatures to kelvin, or return None if they were never recorded.
    """
    temperature_c = np.asarray(temperature_c, dtype=float)
    if temperature_c.size == 0 or np.all(np.isnan(temperature_c) | (temperature_c == 0)):
        return None
    return temperature_c - ABSOLUTE_ZERO_C


def temperature_from_ipasc(temperature_k) -> np.ndarray:
    return np.asarray(temperature_k, dtype=float) + ABSOLUTE_ZERO_C
