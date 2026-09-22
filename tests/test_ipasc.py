#  Copyright (c) Thomas Else 2023-25.
#  License: MIT

"""
Tests for the IPASC consensus format: reading one, writing one, and the IPASC metadata
block embedded in PATATO's own files.
"""

import tempfile
import unittest
import uuid
from pathlib import Path

import numpy as np

N_DET, N_SAMP, N_WL, N_FRAME = 8, 256, 3, 4
WAVELENGTHS_M = np.array([7.00e-7, 8.00e-7, 9.00e-7])
SAMPLING_RATE_HZ = 4.0e7
SPEED_OF_SOUND = 1500.0
TEMPERATURE_K = 298.15
OVERALL_GAIN = 2.0


def _write_synthetic_ipasc_file(path):
    import pacfish as pf
    from pacfish import MetaDatum, MetadataAcquisitionTags as Tags

    class SyntheticAdapter(pf.BaseAdapter):
        def generate_binary_data(self):
            # Fixed seed so the fixture is reproducible.
            return np.random.default_rng(0).normal(size=(N_DET, N_SAMP, N_WL, N_FRAME))

        def generate_device_meta_data(self):
            creator = pf.DeviceMetaDataCreator()
            creator.set_general_information(
                str(uuid.uuid4()), np.array([-0.02, 0.02, 0.0, 0.0, 0.0, 0.04])
            )
            for i in range(N_DET):
                element = pf.DetectionElementCreator()
                element.set_detector_position(
                    np.array([-0.02 + 0.04 * i / (N_DET - 1), 0.0, 0.03])
                )
                creator.add_detection_element(element.get_dictionary())
            return creator.finalize_device_meta_data()

        def set_metadata_value(self, metadatum: MetaDatum):
            return {
                Tags.ACQUISITION_WAVELENGTHS: WAVELENGTHS_M,
                Tags.AD_SAMPLING_RATE: SAMPLING_RATE_HZ,
                Tags.TEMPERATURE_CONTROL: np.full(N_FRAME, TEMPERATURE_K),
                Tags.OVERALL_GAIN: OVERALL_GAIN,
                Tags.SPEED_OF_SOUND: SPEED_OF_SOUND,
            }.get(metadatum)

    pf.write_data(str(path), SyntheticAdapter().generate_pa_data())


class TestIPASCRead(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "synthetic_ipasc.hdf5"
        _write_synthetic_ipasc_file(self.path)

    def tearDown(self):
        self._tmp.cleanup()

    def test_reads_an_ipasc_file_into_patato_units(self):
        from patato import PAData
        from patato.io.ipasc.read_ipasc import IPASCInterface

        pa_data = PAData.from_hdf5(str(self.path))
        reader = pa_data.scan_reader
        self.assertIsInstance(reader, IPASCInterface)
        self.assertIsNone(pa_data.scan_writer)  # no PATATO writer backs an IPASC file

        # IPASC stores [detectors, samples, wavelengths, frames].
        self.assertEqual(pa_data.get_time_series().shape, (N_FRAME, N_WL, N_DET, N_SAMP))
        self.assertEqual(pa_data.get_n_samples(), N_SAMP)
        # Metres to nanometres, kelvin to Celsius.
        np.testing.assert_allclose(pa_data.get_wavelengths(), [700.0, 800.0, 900.0])
        np.testing.assert_allclose(np.asarray(reader.get_temperature()), 25.0, atol=1e-9)
        self.assertEqual(pa_data.get_sampling_frequency(), SAMPLING_RATE_HZ)
        self.assertEqual(pa_data.get_speed_of_sound(), SPEED_OF_SOUND)
        np.testing.assert_allclose(np.asarray(reader.get_correction_factor()), OVERALL_GAIN)

        # A file recording no gain falls back to ones. NaN would propagate through the
        # energy correction into every reconstructed pixel.
        reader.pa_data.meta_data_acquisition.pop("overall_gain")
        np.testing.assert_array_equal(np.asarray(reader.get_correction_factor()), 1.0)


class TestIPASCExport(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.source = Path(__file__).parent / "test_data.hdf5"
        self.out = Path(self._tmp.name) / "exported_ipasc.hdf5"
        from patato import export_to_ipasc

        export_to_ipasc(str(self.source), str(self.out))

    def tearDown(self):
        self._tmp.cleanup()

    def test_export_passes_pacfish_validation(self):
        import pacfish as pf
        from pacfish import MetadataAcquisitionTags, MetadataDeviceTags
        from pacfish.qualitycontrol import ConsistencyChecker

        data = pf.load_data(str(self.out))
        missing = [
            datum.tag
            for datum in MetadataAcquisitionTags.TAGS
            if datum.mandatory and datum.tag not in data.meta_data_acquisition
        ]
        self.assertEqual(missing, [], f"missing minimal acquisition metadata: {missing}")
        np.testing.assert_array_equal(
            data.get_sizes(), np.shape(data.binary_time_series_data)
        )
        for element in data.meta_data_device[MetadataDeviceTags.DETECTORS.tag].values():
            self.assertIn(MetadataDeviceTags.DETECTOR_POSITION.tag, element)

        checker = ConsistencyChecker(verbose=False, log_file_path=None)
        self.assertTrue(checker.check_binary_data(data.binary_time_series_data))
        self.assertTrue(checker.check_acquisition_meta_data(data.meta_data_acquisition))

    def test_round_trip_preserves_the_scan(self):
        from patato import PAData

        source = PAData.from_hdf5(str(self.source))
        exported = PAData.from_hdf5(str(self.out))
        np.testing.assert_array_equal(
            np.asarray(source.get_time_series().raw_data),
            np.asarray(exported.get_time_series().raw_data),
        )
        np.testing.assert_allclose(
            source.get_wavelengths(), exported.get_wavelengths(), rtol=1e-9
        )
        self.assertEqual(
            source.get_sampling_frequency(), exported.get_sampling_frequency()
        )
        np.testing.assert_allclose(
            np.asarray(source.get_scan_geometry()),
            np.asarray(exported.get_scan_geometry()),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(source.get_overall_correction_factor()),
            np.asarray(exported.get_overall_correction_factor()),
        )
        source.scan_reader.close()

    def test_nothing_is_written_that_the_source_does_not_record(self):
        """Absent beats invented: a guessed value is indistinguishable from a measurement."""
        import pacfish as pf
        from pacfish import MetadataAcquisitionTags as Tags
        from pacfish import MetadataDeviceTags

        data = pf.load_data(str(self.out))
        for datum in (
            Tags.SCANNING_METHOD,          # a property of the hardware
            Tags.MEASUREMENTS_PER_IMAGE,   # follows from the scanning method
            Tags.ACOUSTIC_COUPLING_AGENT,  # a property of the setup
            Tags.MEASUREMENT_SPATIAL_POSES,  # IPASC does not define the component order
            Tags.MEASUREMENT_TIMESTAMPS,   # this fixture records no UTC offset
        ):
            self.assertNotIn(datum.tag, data.meta_data_acquisition)

        general = data.meta_data_device[MetadataDeviceTags.GENERAL.tag]
        # The fixture records no serials, and a field of view is never asserted.
        self.assertNotIn(MetadataDeviceTags.UNIQUE_IDENTIFIER.tag, general)
        self.assertNotIn(MetadataDeviceTags.FIELD_OF_VIEW.tag, general)

        for key, value in data.meta_data_acquisition.items():
            array = np.asarray(value)
            if array.dtype.kind == "f":
                self.assertTrue(np.isfinite(array).all(), f"NaN written for {key}")


class TestIPASCBlockInPatatoFiles(unittest.TestCase):
    def test_patato_files_carry_an_ipasc_block_without_becoming_one(self):
        import h5py

        from patato import PAData
        from patato.io.attribute_tags import IPASCTags
        from patato.io.hdf.hdf5_reader_factory import is_ipasc_hdf5

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "resaved.hdf5"
            source = PAData.from_hdf5(str(Path(__file__).parent / "test_data.hdf5"))
            source.save_hdf5(str(out))
            source.scan_reader.close()

            with h5py.File(out, "r") as file:
                meta = file[IPASCTags.GROUP][IPASCTags.META_DATA]
                # Metres, not nanometres, while the PATATO payload keeps its own units.
                self.assertLess(np.max(meta["acquisition_wavelengths"][:]), 1e-5)
                self.assertGreater(np.max(file["wavelengths"][:]), 100.0)
                self.assertIn("raw_data", file)
                # The block must not make the file look like an IPASC container.
                self.assertFalse(is_ipasc_hdf5(file))


if __name__ == "__main__":
    unittest.main()
