import logging

import pydicom


def fix_issues_in_pydicom(dcm: pydicom.FileDataset) -> tuple[pydicom.FileDataset, bool]:
    needed_fix = False

    try:
        if dcm.file_meta.TransferSyntaxUID in ['1.2.840.10008.1.2.4.50']:
            if dcm.SamplesPerPixel == 3:
                if dcm.PhotometricInterpretation not in ['YBR_FULL', 'YBR_FULL_422']:
                    logging.error(f"JPEG Transfer syntax but listed as {dcm.PhotometricInterpretation}")
                    dcm.PhotometricInterpretation = "YBR_FULL_422"
                    needed_fix = True

        elif dcm.file_meta.TransferSyntaxUID in ['1.2.840.10008.1.2.4.70']:
            if dcm.PhotometricInterpretation in ["RGB"]:
                dcm.PhotometricInterpretation = "INSANE" # WE undo a stupid RGB to  YBR
                needed_fix = True

    except Exception:
        logging.exception(f"Error in extracting and fixing PhotometricInterpretation for JPEG")

    return dcm, needed_fix
