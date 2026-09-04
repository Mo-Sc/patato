#  Copyright (c) Thomas Else 2023-25.
#  License: MIT

import numpy as np
import pacfish as pf

from .metadata_mapping import build_ipasc_metadata


class PatatoAdapterToIPASCFormat(pf.BaseAdapter):
    """Converts any PATATO reader into the IPASC representation.

    Metadata come from the shared mapping, so a native IPASC export and the IPASC metadata
    embedded in a PATATO HDF5 file always agree.
    """

    def __init__(self, reader):
        self.reader = reader
        self.acquisition, self.device = build_ipasc_metadata(reader)
        super().__init__()
        # pacfish only queries the standard tag list, so anything left over is a custom
        # parameter and has to be added by hand.
        for key, value in self.acquisition.items():
            if key not in self.pa_data.meta_data_acquisition:
                self.add_custom_meta_datum_field(key, value)

    def generate_binary_data(self) -> np.ndarray:
        # [frames, wavelengths, detectors, samples] -> [detectors, samples, wavelengths, frames]
        time_series = np.asarray(self.reader.get_pa_data().raw_data)
        return np.transpose(time_series, (2, 3, 1, 0))

    def generate_device_meta_data(self) -> dict:
        return self.device

    def set_metadata_value(self, metadatum: pf.MetaDatum) -> object:
        return self.acquisition.get(metadatum.tag)
