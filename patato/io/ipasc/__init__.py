#  Copyright (c) Thomas Else 2023-25.
#  License: MIT

from .read_ipasc import IPASCInterface


def write_ipasc(reader, out_path: str) -> str:
    """Write an open PATATO reader out as a native IPASC raw time series file.

    Only the raw time series and its acquisition metadata are transferred; reconstructions,
    ultrasound and annotations have no representation in the IPASC format, which is scoped to
    raw data by design. PATATO's own HDF5 export keeps those.
    """
    import pacfish as pf

    from .ipasc_export import PatatoAdapterToIPASCFormat

    pf.write_data(out_path, PatatoAdapterToIPASCFormat(reader).generate_pa_data())
    return out_path


def export_to_ipasc(hdf5_path, out_path=None) -> str:
    """Convert a PATATO HDF5 scan into a native IPASC raw time series file."""
    from pathlib import Path

    from ..hdf.hdf5_reader_factory import get_hdf5_reader

    if out_path is None:
        source = Path(hdf5_path)
        out_path = str(source.with_name(f"{source.stem}_ipasc.hdf5"))

    reader = get_hdf5_reader(str(hdf5_path))
    try:
        return write_ipasc(reader, out_path)
    finally:
        reader.close()
