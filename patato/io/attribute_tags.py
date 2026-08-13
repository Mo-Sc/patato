#  Copyright (c) Thomas Else 2023-25.
#  License: MIT
# Convention: attribute tags are uppercase. HDF5 groups or datasets are lowercase.


class PreprocessingAttributeTags:
    HIGH_PASS_FILTER = "FILTER_HIGH_PASS"
    LOW_PASS_FILTER = "FILTER_LOW_PASS"
    TIME_INTERPOLATION = "INTERPOLATE_TIME"
    DETECTOR_INTERPOLATION = "INTERPOLATE_DETECTORS"
    IMPULSE_RESPONSE = "IRF"
    HILBERT_TRANSFORM = "HILBERT_TRANSFORM"
    ENVELOPE_DETECTION = "ENVELOPE_DETECTION"
    WINDOW_SIZE = "WINDOW_SIZE"
    PROCESSING_ALGORITHM = "PREPROCESSING_ALGORITHM"
    COUPLANT_CORRECTION = "COUPLANT_CORRECTION"
    COUPLANT_PATH_LENGTH = "COUPLANT_PATH_LENGTH"


class ReconAttributeTags:
    ADDITIONAL_PARAMETERS = "RECONSTRUCTION_PARAMS"
    X_FIELD_OF_VIEW = "RECONSTRUCTION_FIELD_OF_VIEW_X"
    Y_FIELD_OF_VIEW = "RECONSTRUCTION_FIELD_OF_VIEW_Y"
    Z_FIELD_OF_VIEW = "RECONSTRUCTION_FIELD_OF_VIEW_Z"
    X_NUMBER_OF_PIXELS = "RECONSTRUCTION_NX"
    Y_NUMBER_OF_PIXELS = "RECONSTRUCTION_NY"
    Z_NUMBER_OF_PIXELS = "RECONSTRUCTION_NZ"
    RECONSTRUCTION_ALGORITHM = "RECONSTRUCTION_ALGORITHM"
    SPEED_OF_SOUND = "RECONSTRUCTION_SPEED_OF_SOUND"
    OLD_FIELD_OF_VIEW = "RECON_FOV"
    OLD_RECON_NX = "RECON_NX"


class UnmixingAttributeTags:
    RESOLUTION_REDUCE = "RESOLUTION_REDUCE"
    WAVELENGTH_RANGE = "WAVELENGTH_RANGE"
    WAVELENGTH_INDICES = "WAVELENGTH_INDICES"
    UNMIXING_WAVELENGTHS = "WAVELENGTHS"
    SPECTRA = "SPECTRA"
    COMPUTE_SO2 = "SO2"
    COMPUTE_THB = "THB"
    SUFFIX = "SUFFIX"
    HAEMOGLOBIN = "HB"
    OXYHAEMOGLOBIN = "HBO2"
    MELANIN = "MELANIN"
    ICG = "ICG"


class ROITags:
    Z_POSITION = "Z"
    REPETITION = "REPETITION"
    ROI_NAME = "CLASS"
    ROI_POSITION = "POSITION"
    RUN = "RUN"
    GENERATED_ROI = "GENERATED"
    ROI_TYPE = "SHAPE_TYPE"
    AX0_INDEX = "AX0_INDEX"
    ROI_GROUP = "ROI_GROUP"
    ROI_ID = "ROI_ID"


class GCAttributeTags:
    STEPS = "STEPS"
    BUFFER = "BUFFER"
    SKIP_START = "SKIP_START"


class HDF5Tags:
    POWER = "power"
    OVERALL_CORR = "correction_factor"
    RAW_DATA = "raw_data"
    RECONSTRUCTION = "reconstructions"
    UNMIXED = "unmixed"
    SO2 = "so2"
    THB = "thb"
    SPEED_OF_SOUND = "speedofsound"
    SAMPLING_FREQ = "fs"
    SCAN_GEOMETRY = "geometry"
    WAVELENGTH = "wavelengths"
    IMPULSE_RESPONSE = "impulse_response"
    Z_POSITION = "z-pos"
    REPETITION = "repetition"
    REGIONS_OF_INTEREST = "rois"
    RUN = "run"
    DELTA_SO2 = "dso2"
    BASELINE_SO2 = "baseline_so2"
    BASELINE_SO2_STANDARD_DEVIATION = "baseline_so2_sigma"
    TIMESTAMP = "timestamp"
    TEMPERATURE = "temperature"
    ULTRASOUND_FRAME_OFFSET = "ultrasound-frame-offset"
    DATE = "date"
    ORIGINAL_NAME = "original_name"
    SCAN_COMMENT = "comment"
    WATER_ABSORPTION_COEFF = "water-absorption-coefficients"
    WATER_PATHLENGTH = "pathlength"
    ULTRASOUND = "ultrasounds"
    ULTRASOUND_FIELD_OF_VIEW = "fov"
    SCAN_NAME = "scan_name"
    CLINICAL_METADATA = "clinical_metadata"
    FILE_ORIGIN = "file_origin"
    SEGMENTATION = "seg"
    DELTA_ICG = "dicg"
    BASELINE_ICG = "baseline_icg"
    BASELINE_ICG_SIGMA = "baseline_icg_sigma"
    AXIS0_MEANING = "axis0_meaning"
    AXIS1_MEANING = "axis1_meaning"
    SPECTRA = "spectra"


class AxisNameTags:
    WAVELENGTH = "WAVELENGTH"
    SPECTRA = "SPECTRA"
    PARAM = "PARAMETER"
    FRAME = "FRAME"
    REDUNDANT = "REDUNDANT"  # axis exists only for structural consistency
    UNSPECIFIED = "UNSPECIFIED"  # no information available about axis meaning


# Map axis-1 meaning to the HDF5 attribute key that stores its labels
_AXIS1_HDF5ATTR_MAP = {
    AxisNameTags.WAVELENGTH: HDF5Tags.WAVELENGTH,
    AxisNameTags.SPECTRA: HDF5Tags.SPECTRA,
}
