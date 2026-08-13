import h5py
import numpy as np

from patato.io.attribute_tags import HDF5Tags, ROITags
from patato.io.hdf.hdf5_interface import HDF5Reader, HDF5Writer
from patato.utils.rois.roi_type import ROI


def _scan_file(path, n_frames=4):
    with h5py.File(path, "w") as file:
        file.create_dataset(HDF5Tags.RAW_DATA, data=np.zeros((n_frames, 1, 2, 2)))


def test_explicit_roi_frames_and_ids_round_trip(tmp_path):
    path = tmp_path / "scan.h5"
    _scan_file(path)

    writer = HDF5Writer(path)
    writer.add_roi(
        ROI(
            [[0.0, 0.0], [1.0, 1.0]],
            0,
            0,
            0,
            "PATARI",
            "undefined",
            True,
            [2, 3],
            "polygon",
            8,
            9,
        ),
        generated=True,
    )
    writer.file.close()

    reader = HDF5Reader(path)
    roi = next(iter(reader._get_rois().values()))
    assert roi.ax0_index.tolist() == [2, 3]
    assert roi.roi_group_id == 8
    assert roi.roi_id == 9
    reader.file.close()

