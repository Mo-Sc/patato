#  Copyright (c) Thomas Else 2023-25.
#  License: MIT

"""
Round trip tests for the IPASC consensus format.

A synthetic IPASC file is generated with pacfish and read back through PATATO, checking that
the acquisition metadata survive the conversion into PATATO's internal units.
"""

import tempfile
import unittest
import uuid
from pathlib import Path

import numpy as np

N_DET, N_SAMP, N_WL, N_FRAME = 8, 256, 3, 4
WAVELENGTHS_M = np.array([7.00e-7, 8.00e-7, 9.00e-7])
FIRST_TIMESTAMP_S = 1657124349.46  # 2022-07-06T16:19:09Z
SAMPLING_RATE_HZ = 4.0e7
SPEED_OF_SOUND = 1500.0
TEMPERATURE_K = 298.15
OVERALL_GAIN = 2.0
Z_POSITION_M = 0.001


def _write_synthetic_ipasc_file(path):
    import pacfish as pf
    from pacfish import MetaDatum, MetadataAcquisitionTags as Tags

    class SyntheticAdapter(pf.BaseAdapter):
        def generate_binary_data(self):
            # Fixed seed so the fixture is reproducible.
            return np.random.default_rng(0).normal(
                size=(N_DET, N_SAMP, N_WL, N_FRAME)
            )

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
                Tags.MEASUREMENT_TIMESTAMPS: FIRST_TIMESTAMP_S
                + np.arange(N_FRAME) * 0.1,
                Tags.TEMPERATURE_CONTROL: np.full(N_FRAME, TEMPERATURE_K),
                Tags.OVERALL_GAIN: OVERALL_GAIN,
                Tags.SPEED_OF_SOUND: SPEED_OF_SOUND,
                Tags.MEASUREMENT_SPATIAL_POSES: np.tile(
                    np.array([0.0, Z_POSITION_M, 0.0, 0.0, 0.0, 0.0]), (N_FRAME, 1)
                ),
                Tags.SCANNING_METHOD: "full_scan",
                Tags.ACOUSTIC_COUPLING_AGENT: "H2O",
            }.get(metadatum)

    pf.write_data(str(path), SyntheticAdapter().generate_pa_data())


class TestIPASC(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "synthetic_ipasc.hdf5"
        _write_synthetic_ipasc_file(self.path)

    def tearDown(self):
        self._tmp.cleanup()

    def test_ipasc_file_is_detected(self):
        import h5py

        from patato.io.hdf.hdf5_reader_factory import is_ipasc_hdf5

        with h5py.File(self.path, "r") as file:
            self.assertTrue(is_ipasc_hdf5(file))

        patato_file = Path(__file__).parent / "test_data.hdf5"
        with h5py.File(patato_file, "r") as file:
            self.assertFalse(is_ipasc_hdf5(file))

    def test_reader_is_selected_and_data_transposed(self):
        from patato import PAData
        from patato.io.ipasc.read_ipasc import IPASCInterface

        pa_data = PAData.from_hdf5(str(self.path))
        self.assertIsInstance(pa_data.scan_reader, IPASCInterface)
        # IPASC files have no PATATO writer backing them.
        self.assertIsNone(pa_data.scan_writer)
        # IPASC stores [detectors, samples, wavelengths, frames].
        self.assertEqual(
            pa_data.get_time_series().shape, (N_FRAME, N_WL, N_DET, N_SAMP)
        )
        self.assertEqual(pa_data.get_n_samples(), N_SAMP)

    def test_metadata_are_converted_to_patato_units(self):
        from patato import PAData

        pa_data = PAData.from_hdf5(str(self.path))
        reader = pa_data.scan_reader

        # IPASC works in metres, PATATO in nanometres.
        np.testing.assert_allclose(
            pa_data.get_wavelengths(), [700.0, 800.0, 900.0], rtol=1e-9
        )
        self.assertEqual(pa_data.get_sampling_frequency(), SAMPLING_RATE_HZ)
        self.assertEqual(pa_data.get_speed_of_sound(), SPEED_OF_SOUND)

        # IPASC timestamps are POSIX seconds; PATATO counts from year 1.
        self.assertEqual(pa_data.get_scan_datetime().year, 2022)
        timestamps = np.asarray(pa_data.get_timestamps())
        self.assertEqual(timestamps.shape, (N_FRAME, N_WL))
        np.testing.assert_allclose(
            timestamps[0, 0], FIRST_TIMESTAMP_S + 62135596800.0
        )

        # IPASC reports kelvin, PATATO degrees Celsius.
        np.testing.assert_allclose(
            np.asarray(reader.get_temperature()), 25.0, atol=1e-9
        )
        np.testing.assert_allclose(
            np.asarray(reader.get_correction_factor()), OVERALL_GAIN
        )

        geometry = np.asarray(pa_data.get_scan_geometry())
        self.assertEqual(geometry.shape, (N_DET, 3))
        np.testing.assert_allclose(geometry[:, 0].min(), -0.02)
        np.testing.assert_allclose(geometry[:, 0].max(), 0.02)

    def test_spatial_poses_are_not_mapped_to_an_axis(self):
        """IPASC does not define the order of the six pose components.

        Picking one and calling it the scanner axis would be a guess, so the position is
        reported as unknown instead.
        """
        from patato import PAData

        reader = PAData.from_hdf5(str(self.path)).scan_reader
        self.assertTrue(np.all(np.isnan(np.asarray(reader.get_scanner_z_position()))))

    def test_run_and_repetition_numbers_are_not_invented(self):
        """IPASC has no notion of runs or repetitions, so both are reported as unknown."""
        from patato import PAData

        reader = PAData.from_hdf5(str(self.path)).scan_reader
        np.testing.assert_array_equal(np.asarray(reader.get_run_numbers()), 0)
        np.testing.assert_array_equal(np.asarray(reader.get_repetition_numbers()), 0)

    def test_absent_content_uses_empty_containers(self):
        from patato import PAData

        pa_data = PAData.from_hdf5(str(self.path))
        # Reconstructions, ultrasound and annotations do not exist in the IPASC format.
        self.assertEqual(pa_data.get_scan_reconstructions(), {})
        self.assertEqual(pa_data.get_ultrasound(), {})
        self.assertEqual(pa_data.get_rois(), {})
        self.assertIsNone(pa_data.scan_reader.get_segmentation())
        self.assertIsNone(pa_data.get_impulse_response())


if __name__ == "__main__":
    unittest.main()


class TestIPASCExport(unittest.TestCase):
    """Export a PATATO scan into the IPASC format and check it against pacfish."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.source = Path(__file__).parent / "test_data.hdf5"
        self.out = Path(self._tmp.name) / "exported_ipasc.hdf5"

    def tearDown(self):
        self._tmp.cleanup()

    def test_export_contains_all_minimal_metadata(self):
        import pacfish as pf
        from pacfish import MetadataAcquisitionTags, MetadataDeviceTags

        from patato import export_to_ipasc

        export_to_ipasc(str(self.source), str(self.out))
        data = pf.load_data(str(self.out))

        missing = [
            datum.tag
            for datum in MetadataAcquisitionTags.TAGS
            if datum.mandatory and datum.tag not in data.meta_data_acquisition
        ]
        self.assertEqual(missing, [], f"missing minimal acquisition metadata: {missing}")
        # Device minimal metadata are covered by test_device_identity_is_not_invented.

        detectors = data.meta_data_device[MetadataDeviceTags.DETECTORS.tag]
        self.assertGreater(len(detectors), 0)
        for element in detectors.values():
            self.assertIn(MetadataDeviceTags.DETECTOR_POSITION.tag, element)

    def test_device_identity_comes_from_the_source_or_is_absent(self):
        """The device UUID is derived from recorded serials, never invented.

        The fixture records no hardware identifiers, so no device UUID is written. A field
        of view is never written: IPASC defines it as the volume the device can detect, and
        every value a scan records is a grid someone chose.
        """
        import pacfish as pf
        from pacfish import MetadataDeviceTags

        from patato import export_to_ipasc

        export_to_ipasc(str(self.source), str(self.out))
        general = pf.load_data(str(self.out)).meta_data_device[
            MetadataDeviceTags.GENERAL.tag
        ]
        self.assertNotIn(MetadataDeviceTags.UNIQUE_IDENTIFIER.tag, general)
        self.assertNotIn(MetadataDeviceTags.FIELD_OF_VIEW.tag, general)

    def test_device_uuid_is_derived_from_recorded_serials(self):
        """Identical identifiers give the same UUID; a changed serial gives a different one."""
        from patato.io.ipasc.metadata_mapping import _device_uuid

        recorded = {"DeviceSN": "2-21-03", "DAQ_MACaddress": "00:11:1c:04:3a:f0"}
        self.assertEqual(_device_uuid(recorded), _device_uuid(dict(recorded)))
        self.assertNotEqual(
            _device_uuid(recorded), _device_uuid({**recorded, "DeviceSN": "1-14-02"})
        )
        # A software update must not make the scanner look like a different device.
        self.assertEqual(
            _device_uuid(recorded), _device_uuid({**recorded, "SW_Version": "9.9.9"})
        )
        self.assertIsNone(_device_uuid({}))

    def test_device_properties_are_not_asserted(self):
        """Scanning method and coupling agent are properties of the hardware and setup."""
        import pacfish as pf
        from pacfish import MetadataAcquisitionTags as Tags

        from patato import export_to_ipasc

        export_to_ipasc(str(self.source), str(self.out))
        acquisition = pf.load_data(str(self.out)).meta_data_acquisition
        for datum in (
            Tags.SCANNING_METHOD,
            Tags.MEASUREMENTS_PER_IMAGE,
            Tags.ACOUSTIC_COUPLING_AGENT,
        ):
            self.assertNotIn(datum.tag, acquisition)

    def test_export_passes_consistency_checks(self):
        import pacfish as pf
        from pacfish.qualitycontrol import ConsistencyChecker

        from patato import export_to_ipasc

        export_to_ipasc(str(self.source), str(self.out))
        data = pf.load_data(str(self.out))

        checker = ConsistencyChecker(verbose=False, log_file_path=None)
        self.assertTrue(checker.check_binary_data(data.binary_time_series_data))
        self.assertTrue(checker.check_acquisition_meta_data(data.meta_data_acquisition))

    def test_export_declares_the_shape_it_wrote(self):
        import pacfish as pf

        from patato import export_to_ipasc

        export_to_ipasc(str(self.source), str(self.out))
        data = pf.load_data(str(self.out))
        np.testing.assert_array_equal(
            data.get_sizes(), np.shape(data.binary_time_series_data)
        )

    def test_round_trip_preserves_raw_data_and_metadata(self):
        from patato import PAData, export_to_ipasc

        source = PAData.from_hdf5(str(self.source))
        export_to_ipasc(str(self.source), str(self.out))
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
        source.scan_reader.close()

    def test_relative_timestamps_are_omitted_rather_than_written_negative(self):
        import pacfish as pf
        from pacfish import MetadataAcquisitionTags

        from patato import export_to_ipasc

        # This fixture counts acquisition time from the start of the scan, not from an
        # absolute epoch, so it cannot be expressed as an IPASC POSIX timestamp.
        export_to_ipasc(str(self.source), str(self.out))
        data = pf.load_data(str(self.out))
        self.assertNotIn(
            MetadataAcquisitionTags.MEASUREMENT_TIMESTAMPS.tag,
            data.meta_data_acquisition,
        )

    def test_spatial_poses_are_omitted_because_the_unit_is_unverified(self):
        import pacfish as pf
        from pacfish import MetadataAcquisitionTags

        from patato import export_to_ipasc

        export_to_ipasc(str(self.source), str(self.out))
        data = pf.load_data(str(self.out))
        self.assertNotIn(
            MetadataAcquisitionTags.MEASUREMENT_SPATIAL_POSES.tag,
            data.meta_data_acquisition,
        )


class TestIPASCMetadataInPatatoFiles(unittest.TestCase):
    """PATATO's own HDF5 files carry the IPASC metadata blocks alongside their payload."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.out = Path(self._tmp.name) / "resaved.hdf5"
        from patato import PAData

        source = PAData.from_hdf5(str(Path(__file__).parent / "test_data.hdf5"))
        source.save_hdf5(str(self.out))
        source.scan_reader.close()

    def tearDown(self):
        self._tmp.cleanup()

    def test_metadata_groups_are_written_in_si_units(self):
        import h5py

        from patato.io.attribute_tags import IPASCTags

        with h5py.File(self.out, "r") as file:
            self.assertIn(IPASCTags.GROUP, file)
            ipasc = file[IPASCTags.GROUP]
            self.assertIn(IPASCTags.META_DATA, ipasc)
            self.assertIn(IPASCTags.META_DATA_DEVICE, ipasc)
            meta = ipasc[IPASCTags.META_DATA]
            # PATATO stores nanometres internally, IPASC requires metres.
            self.assertLess(np.max(meta["acquisition_wavelengths"][:]), 1e-5)
            # Timestamps, when absolute, are seconds since the Unix epoch not the .NET epoch.
            if "measurement_timestamps" in meta:
                self.assertLess(np.max(meta["measurement_timestamps"][:]), 4e9)
                self.assertGreater(np.min(meta["measurement_timestamps"][:]), 0.0)
            # The PATATO payload is untouched.
            self.assertIn("raw_data", file)
            self.assertIn("wavelengths", file)
            self.assertGreater(np.max(file["wavelengths"][:]), 100.0)

    def test_patato_file_is_not_mistaken_for_an_ipasc_file(self):
        import h5py

        from patato.io.hdf.hdf5_reader_factory import is_ipasc_hdf5

        with h5py.File(self.out, "r") as file:
            self.assertFalse(is_ipasc_hdf5(file))
