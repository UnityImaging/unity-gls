import logging
import highdicom
import numpy as np

from .image import apply_frame_conversion
from .time import work_out_frame_time


class DICOMFileReader(highdicom.io.ImageFileReader):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self._raw_frame_cache = None
        self._corrected_frame_cache = None
        self._frame_time = None

    def read_frame_corrected(self, index: int, correct_color: bool = True) -> np.ndarray:
        raw_frame = self.read_frame(index=index, correct_color=correct_color)
        correct_frame = apply_frame_conversion(raw_frame, metadata=self.metadata)
        return correct_frame

    def read_frame_corrected_cached(self, index: int, correct_color: bool = True) -> np.ndarray:
        if self._corrected_frame_cache is None:
            self._corrected_frame_cache = [None] * self.number_of_frames

        if self._corrected_frame_cache[index] is None:
            self._corrected_frame_cache[index] = self.read_frame_corrected(index=index, correct_color=correct_color)

        return self._corrected_frame_cache[index]

    @property
    def frame_time(self):
        if self._frame_time is None:
            self._frame_time = work_out_frame_time(self.metadata, default=33)

        return self._frame_time

    @property
    def num_frames(self):
        return self.metadata.get('NumberOfFrames', 1)

    @property
    def width(self):
        return self.metadata.get('Columns')

    @property
    def height(self):
        return self.metadata.get('Rows')

    @property
    def all_frame_corrected_cached(self):
        if self._corrected_frame_cache is None:
            return False
        else:
            # This is insane as cannot do all on list of np arrays if ndim > 1
            return all([x is not None for x in self._corrected_frame_cache])

