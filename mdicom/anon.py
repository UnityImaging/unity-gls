import logging

import pydicom
import pydicom.uid

from typing import Optional


def make_anon_echo_dicom_with_new_pixel_data(pixel_data, source_dcm: Optional[pydicom.FileDataset]=None):

    if pixel_data.ndim != 4:
        logging.error("Expected pixel_data to be dim=4 frame,row,col,samples")
        raise Exception("Expected pixel_data to be dim=4 frame,row,col,samples")

    NumberOfFrames, Rows, Columns, SamplesPerPixel = pixel_data.shape

    if SamplesPerPixel == 3:
        PhotometricInterpretation = "RGB"
        PlanarConfiguration = 0
    elif SamplesPerPixel == 1:
        PhotometricInterpretation = "MONOCHROME2"
        PlanarConfiguration = None
    else:
        logging.error("Expected 1 or 3 samples per pixel")
        raise Exception("Expected 1 or 3 samples per pixel")

    if source_dcm is None:
        FrameTime = 33
    else:
        FrameTime = source_dcm.get('FrameTime', 33)

    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.3.1'
    file_meta.MediaStorageSOPInstanceUID = "1.2.3"
    file_meta.ImplementationClassUID = "1.2.3.4"
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    ds = pydicom.dataset.FileDataset(filename_or_obj=None,
                                     dataset={},
                                     file_meta=file_meta,
                                     is_implicit_VR=False,
                                     is_little_endian=True)

    ds.Rows = Rows
    ds.Columns = Columns
    ds.SamplesPerPixel = SamplesPerPixel
    ds.PhotometricInterpretation = PhotometricInterpretation
    ds.PixelRepresentation = 0
    ds.PlanarConfiguration = PlanarConfiguration
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.NumberOfFrames = NumberOfFrames
    ds.FrameTime = FrameTime

    ds.PixelData = pixel_data.tobytes()

    ds.compress(pydicom.uid.RLELossless)

    return ds
