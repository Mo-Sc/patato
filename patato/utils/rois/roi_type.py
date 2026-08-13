#  Copyright (c) Thomas Else 2023-25.
#  License: MIT

from typing import TYPE_CHECKING

import numpy as np

from ...io.attribute_tags import ROITags
from ...utils.mask_operations import generate_mask

if TYPE_CHECKING:
    from typing import Dict
    from ...core.image_structures.image_sequence import ImageSequence


class ROI:
    """Class to store regions of interests."""

    def get_area(self):
        return self.get_polygon().area

    def get_polygon(self):
        from shapely.geometry import Polygon, MultiPolygon
        from shapely.validation import make_valid

        if type(self.points) in [Polygon, MultiPolygon]:
            if not self.points.is_valid:
                return make_valid(self.points)
            else:
                return self.points
        else:
            return make_valid(Polygon(self.points))

    @property
    def attributes(self) -> "Dict":
        output = {
            ROITags.Z_POSITION: self.z,
            ROITags.RUN: self.run,
            ROITags.REPETITION: self.repetition,
            ROITags.ROI_NAME: self.roi_class,
            ROITags.ROI_POSITION: self.position,
            ROITags.GENERATED_ROI: self.generated,
            ROITags.ROI_TYPE: self.shape_type,
        }
        if self.ax0_index.size:
            output[ROITags.AX0_INDEX] = self.ax0_index
        if self.roi_group_id is not None:
            output[ROITags.ROI_GROUP] = self.roi_group_id
        if self.roi_id is not None:
            output[ROITags.ROI_ID] = self.roi_id
        return output

    def __init__(
        self,
        points,
        z_position,
        run,
        repetition,
        roi_class,
        position,
        generated=False,
        ax0_index=None,
        shape_type: str = "polygon",
        roi_group_id=None,
        roi_id=None,
    ):
        self.points = points
        self.z = z_position
        self.run = run
        self.repetition = repetition
        self.roi_class = roi_class
        self.position = position
        self.generated = generated
        self.ax0_index = np.asarray(
            [] if ax0_index is None else ax0_index, dtype=int
        ).reshape(-1) # Keep frame indices 1D integer arrays for indexing.
        self.shape_type = shape_type
        self.roi_group_id = roi_group_id
        self.roi_id = roi_id

    @classmethod
    def from_polygon_mm(
        cls,
        verts_yx_mm: np.ndarray,
        fov: tuple,
        *,
        z_position: float = 0.0,
        run: float = 0.0,
        repetition: float = 0.0,
        ax0_index: np.ndarray = None,
        roi_class: str = "user",
        position: str = "0",
        generated: bool = True,
        shape_type: str = "polygon",
    ) -> "ROI":
        """Create a ROI from napari world-space vertices (y, x) in mm.

        Converts from napari display coordinates — ``(y_mm, x_mm)``, top-left
        origin, y flipped relative to PATATO — to PATATO physical coordinates
        ``(x_m, y_m)``, centred at the image midpoint, in metres.

        Parameters
        ----------
        verts_yx_mm : np.ndarray of shape (N, 2)
            Polygon vertices in napari world coordinates ``(y_mm, x_mm)``.
        fov : tuple
            ``(fov_x_m, fov_y_m)`` field of view in metres, as returned by
            ``ImageSequence.fov``.
        z_position : float
            Scanner z-position for this ROI (metres).  Used for frame matching
            when the ROI is read back.
        run : float
            Acquisition run number.
        repetition : float
            Acquisition repetition number.
        ax0_index : np.ndarray or None
            Acquisition frame indices this ROI applies to.
        roi_class : str
            Semantic class label, e.g. ``"user"`` or ``"tumour"``.
        position : str
            Position label, e.g. ``"0"`` or ``"left"``.
        generated : bool
            Whether the ROI was auto-generated (``True`` for plugin-created ROIs).
        """
        fov_x_m, fov_y_m = float(fov[0]), float(fov[1])
        arr = np.asarray(verts_yx_mm, dtype=float)
        y_mm = arr[:, 0]
        x_mm = arr[:, 1]
        # napari (top-left, y↓, mm)  →  PATATO (centred, x→, y↑, metres)
        patato_x = x_mm / 1000.0 - fov_x_m / 2.0
        patato_y = fov_y_m / 2.0 - y_mm / 1000.0
        points = np.stack([patato_x, patato_y], axis=1)  # (N, 2): (x, y)
        return cls(
            points=points,
            z_position=z_position,
            run=run,
            repetition=repetition,
            roi_class=roi_class,
            position=position,
            generated=generated,
            ax0_index=ax0_index,
            shape_type=shape_type,
        )

    def to_mask_slice(self, image: "ImageSequence", return_selection=False):
        mask = generate_mask(
            self.points,
            image.fov[0],
            image.shape_2d[-1],
            image.fov[1],
            image.shape_2d[-2],
        )
        mask = mask.reshape(image.shape[-image.n_im_dim :])
        selection = slice(None, None)
        selection = np.where(
            self.ax0_index[None, :] == np.atleast_1d(image.ax_0_labels)[:, None]
        )[0]
        ret_image = image[selection]

        if not return_selection:
            return mask, ret_image
        else:
            return mask, ret_image, selection

    def plot(self, ax=None, **kwargs):
        from ..roi_operations import REGION_COLOUR_MAP, close_loop
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()
        if type(self.points) is np.ndarray:
            plot = ax.plot(
                close_loop(self.points)[:, 0],
                close_loop(self.points)[:, 1],
                label=self.roi_class + "_" + self.position,
                c=REGION_COLOUR_MAP[self.roi_class],
                **kwargs,
            )
        else:
            x, y = self.points.exterior.coords.xy
            plot = ax.plot(
                x,
                y,
                label=self.roi_class + "_" + self.position,
                c=REGION_COLOUR_MAP[self.roi_class],
                **kwargs,
            )
            for interior in self.points.interiors:
                x, y = interior.coords.xy
                plot.append(
                    ax.plot(
                        x,
                        y,
                        label=self.roi_class + "_" + self.position,
                        c=REGION_COLOUR_MAP[self.roi_class],
                        **kwargs,
                    )
                )
        return plot
