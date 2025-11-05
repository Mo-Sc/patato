"""Defines a datatype for datasets that have been processed to only have one value per scan per pixel."""

#  Copyright (c) Thomas Else 2023-25.
#  License: MIT

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    import h5py

from ...core.image_structures.image_sequence import ImageSequence
from ...io.attribute_tags import AxisNameTags, HDF5Tags


class SingleImage(ImageSequence):
    """SingleImage is the datastructure for images like delta sO2.."""

    save_output = True
    axis1_label_tag = AxisNameTags.REDUNDANT

    def get_hdf5_group_name(self):
        labels = self.ax_1_labels
        if labels.size == 1:
            return str(labels.item())
        # Fallback to a stable attribute or generic name
        return str(self.da.attrs.get(HDF5Tags.ORIGINAL_NAME, "single_image"))

    @staticmethod
    def is_single_instance():
        return False
