import logging
import zipfile
import pydicom

from io import BytesIO


def convert_pydicom_to_bytes(dcm: pydicom.FileDataset) -> bytes:
    dicom_buffer = BytesIO()
    pydicom.filewriter.dcmwrite(dicom_buffer, dcm, write_like_original=False)
    return dicom_buffer.getvalue()


def zip_results_dcm_to_bytes(dicom_bytes: bytes) -> bytes:
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, mode='w', compression=zipfile.ZIP_BZIP2, compresslevel=5) as zip_f:
        zip_f.writestr("result.dcm", data=dicom_bytes)

    return zip_buffer.getvalue()
