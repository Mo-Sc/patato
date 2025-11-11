#  Copyright (c) Thomas Else 2023-25.
#  License: MIT

from ...core.image_structures.image_sequence import ImageSequence
from ...io.attribute_tags import HDF5Tags, AxisNameTags


class Ultrasound(ImageSequence):
    """Data structure for reconstructed ultrasound images."""

    save_output = True
    axis1_label_tag = AxisNameTags.REDUNDANT

    @staticmethod
    def is_single_instance():
        return False

    def get_hdf5_group_name(self) -> str:
        return HDF5Tags.ULTRASOUND

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
