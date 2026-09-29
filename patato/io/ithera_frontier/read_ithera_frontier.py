"""
PATATO scan reader for Frontier data, built on top of `custom_python_libraries`.

One PATATO "scan" == one Frontier capture (identified by source + session + GUID).

IMPORTANT: import order. `custom_python_libraries` boots the .NET runtime (pythonnet),
so import it (or this module, which imports it lazily inside `from_db`) before
anything else that might load CLR.

Quick start
-----------
    from frontier_patato_reader import FrontierPatatoReader, FrontierSettings

    settings = FrontierSettings(
        sampling_frequency=40e6,        # Hz   <-- CHECK: not exposed by your loader
        speed_of_sound=1500.0,          # m/s  <-- CHECK
        geometry=my_geometry_array,     # (n_channels, 3) metres  <-- REQUIRED
    )
    FrontierPatatoReader.list_captures(r"D:\\frontier_db")            # find GUIDs
    reader = FrontierPatatoReader.from_db(
        r"D:\\frontier_db", source="patient", session=1,
        capture_guid="0772cd1c-...", settings=settings)
    import patato as pat
    pa = pat.PAData(reader)
    pa.set_default_recon()
    pa.get_scan_reconstructions().imshow()
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from patato.io.hdf.fileimporter import ReaderInterface


@dataclass
class FrontierSettings:
    """Acquisition parameters your loader does not currently return.
    Find them in `capture.MasterMeta` (see inspect_frontier_meta.py) or hardcode
    from the device spec."""
    sampling_frequency: Optional[float] = None   # Hz
    speed_of_sound: Optional[float] = None       # m/s
    geometry: Optional[np.ndarray] = None        # (n_channels, 3) in metres
    impulse_response: Optional[np.ndarray] = None
    drop_channels: Sequence[int] = ()            # e.g. broken channels (not applied unless set)


class FrontierPatatoReader(ReaderInterface):
    def __init__(
        self,
        icapture,
        settings: FrontierSettings,
        name: str = "",
        raw_generator_cls=None,
        recon_generator_cls=None,
        include_recon: bool = False,
    ):
        ReaderInterface.__init__(self)
        self.icapture = icapture
        self.settings = settings
        self.name = name or str(getattr(icapture, "Name", "Frontier scan"))
        self.include_recon = include_recon

        # Lazy import so this module can be imported/tested without .NET installed.
        if raw_generator_cls is None or (include_recon and recon_generator_cls is None):
            from custom_python_libraries.dataloader.frame_generators import (
                OARawSweepGenerator, OAReconSweepGenerator)
            raw_generator_cls = raw_generator_cls or OARawSweepGenerator
            recon_generator_cls = recon_generator_cls or OAReconSweepGenerator
        self._raw_cls = raw_generator_cls
        self._recon_cls = recon_generator_cls
        self._raw_gen = None
        self._ts = None          # cached (sweeps, wl, channels, samples)
        self._timestamps = None  # (sweeps, wl)

    # ------------------------------------------------------------ constructors
    @classmethod
    def from_db(cls, db_path, source, session, capture_guid, settings, **kw):
        """
        source:  'patient'     -> session = patient id (int)
                 'calibration' -> session = session start string 'yyyy-MM-ddTHH:mm:ss'
        """
        from custom_python_libraries import FrontierReader
        db = FrontierReader(db_path=db_path)._capture_db
        sess = cls._get_session(db, source, session)
        for cid in sess.EnumCaptures().Result:
            if cid.get_Guid().ToString() == capture_guid:
                return cls(sess.GetCapture(cid).Result, settings, **kw)
        raise KeyError(f"Capture {capture_guid} not found in {source} {session}")

    @staticmethod
    def _get_session(db, source, session):
        if source == "patient":
            return db.GetPatient(session).Result
        if source == "calibration":
            for s in db.EnumCalibrations().Result:
                if str(s.SessionStart.ToString("yyyy-MM-ddTHH:mm:ss")) == str(session):
                    return s
            raise KeyError(f"Calibration session {session} not found")
        raise ValueError("source must be 'patient' or 'calibration'")

    @classmethod
    def list_captures(cls, db_path, patient_ids: Sequence[int] = (), calibrations=True):
        """Print (source, session, guid, name) for every capture, to pick from."""
        from custom_python_libraries import FrontierReader
        db = FrontierReader(db_path=db_path)._capture_db
        rows = []
        for pid in patient_ids:
            s = db.GetPatient(pid).Result
            for cid in s.EnumCaptures().Result:
                rows.append(("patient", pid, cid.get_Guid().ToString(),
                             s.GetCapture(cid).Result.Name))
        if calibrations:
            for s in db.EnumCalibrations().Result:
                key = str(s.SessionStart.ToString("yyyy-MM-ddTHH:mm:ss"))
                for cid in s.EnumCaptures().Result:
                    rows.append(("calibration", key, cid.get_Guid().ToString(),
                                 s.GetCapture(cid).Result.Name))
        for r in rows:
            print(r)
        return rows

    # ---------------------------------------------------------------- raw data
    def _load(self):
        if self._ts is not None:
            return
        gen = self._raw_cls(self.icapture)
        self._raw_gen = gen
        sweeps, stamps = [], []
        for sweep in gen:                       # sweep: object array of OARawFrame per wavelength
            sweeps.append(np.stack([np.asarray(f) for f in sweep]))   # (wl, samples, channels)
            stamps.append([f.timestamp for f in sweep])
        ts = np.stack(sweeps)                   # (sweeps, wl, samples, channels)
        ts = np.transpose(ts, (0, 1, 3, 2))     # -> (sweeps, wl, channels, samples) as PATATO expects
        if len(self.settings.drop_channels):
            keep = [i for i in range(ts.shape[2]) if i not in set(self.settings.drop_channels)]
            ts = ts[:, :, keep, :]
        self._ts = np.ascontiguousarray(ts)
        self._timestamps = stamps

    def _get_pa_data(self):
        self._load()
        return self._ts, {"fs": self._get_sampling_frequency(),
                          "speedofsound": self.get_speed_of_sound()}

    def _get_wavelengths(self):
        self._load()
        return np.array(self._raw_gen.wavelengths, dtype=float)

    # ---------------------------------------------------------- acquisition info
    def _require(self, value, what):
        if value is None:
            raise NotImplementedError(
                f"{what} is not provided by the Frontier loader. Pass it via "
                f"FrontierSettings (see inspect_frontier_meta.py to look for it in MasterMeta).")
        return value

    def _get_sampling_frequency(self):
        return float(self._require(self.settings.sampling_frequency, "sampling_frequency"))

    def get_speed_of_sound(self):
        return float(self._require(self.settings.speed_of_sound, "speed_of_sound"))

    def _get_sensor_geometry(self):
        geo = np.asarray(self._require(self.settings.geometry, "geometry"), dtype=float)
        if len(self.settings.drop_channels):
            keep = [i for i in range(len(geo)) if i not in set(self.settings.drop_channels)]
            geo = geo[keep]
        return geo

    def get_impulse_response(self):
        return self.settings.impulse_response

    # ------------------------------------------------------------------ metadata
    def get_scan_name(self):
        return self.name

    def get_scan_datetime(self):
        self._load()
        try:
            return str(self._timestamps[0][0])
        except Exception:
            return ""

    def get_scan_comment(self):
        return ""

    def _get_run_numbers(self):
        # Assumption: one "run" per sweep (matches the tutorial's demo layout (frames, wavelengths)).
        self._load()
        n, w = self._ts.shape[:2]
        return np.repeat(np.arange(n)[:, None], w, axis=1)

    def _get_repetition_numbers(self):
        return None

    def _get_scan_times(self):
        return None  # TODO: convert DAQTime (self._timestamps) to seconds once its type/unit is confirmed

    # ------------------------------------------------------------------ optional
    def _get_rois(self):
        return {}

    def _get_temperature(self):
        return None

    def _get_correction_factor(self):
        return None  # note: raw laser energy is a 1.0 placeholder in your loader

    def _get_scanner_z_position(self):
        return None

    def _get_water_absorption(self):
        return None, None

    def _get_segmentation(self):
        return None

    def get_us_data(self):
        return None, None

    def get_us_offsets(self):
        return None

    def _get_datasets(self):
        """Optionally expose Frontier's own on-device OA reconstructions (include_recon=True).
        UNVERIFIED: axis order/orientation of the image array vs PATATO - check visually."""
        output = {"recons": {}}
        if not self.include_recon:
            return output
        import patato as pat
        gen = self._recon_cls(self.icapture)
        data = np.stack([np.stack([np.asarray(f) for f in sweep]) for sweep in gen])  # (sweeps, wl, x, y)
        data = data[:, :, None, :, :]                                                 # (sweeps, wl, 1, x, y)
        fov = gen.fov
        attrs = {"RECONSTRUCTION_ALGORITHM": "Frontier onboard",
                 "RECONSTRUCTION_NX": data.shape[-2], "RECONSTRUCTION_NY": data.shape[-1],
                 "RECONSTRUCTION_NZ": 1,
                 "wavelengths": np.array(gen.wavelengths, dtype=float)}
        image = pat.Reconstruction(
            data, np.array(gen.wavelengths, dtype=float), attributes=attrs,
            field_of_view=[fov.extent_x, fov.extent_y, 0.0])
        output["recons"][("Frontier onboard", "0")] = image
        return output
