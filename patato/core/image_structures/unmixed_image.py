"""This defines the data structure for unmixed datasets."""

#  Copyright (c) Thomas Else 2023-25.
#  License: MIT

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import h5py
    import numpy as np

from ...core.image_structures.image_sequence import ImageSequence
from ...io.attribute_tags import HDF5Tags, AxisNameTags


class UnmixedData(ImageSequence):
    """UnmixedData stores unmixed datasets."""

    save_output = True
    axis1_label_tag = AxisNameTags.SPECTRA

    @staticmethod
    def is_single_instance():
        return False

    def get_hdf5_group_name(self):
        return HDF5Tags.UNMIXED

    @property
    def spectra(self):
        return self.ax_1_labels
