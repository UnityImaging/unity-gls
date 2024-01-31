import math
import random
from collections import namedtuple

import logging
import json

import torch
import numpy as np

import pydicom
import pydicom.pixel_data_handlers

from torch.utils.data import Dataset

from Scantensus.datasets.securion import SecurionSource
from Scantensus.utils.heatmaps import make_curve_labels
from ScantensusPT.transforms.helpers import get_random_transform_parm, get_affine_matrix
from ScantensusPT.utils.heatmaps import render_gaussian_dot_u, render_gaussian_curve_u

UnityInferSet3dT = namedtuple("UnityInferSet3dT", "image unity_i_code label_height_shift label_width_shift")
UnityTrainSet3dT = namedtuple("UnityTrainSet3dT", "image unity_f_code label_data label_height_shift label_width_shift transform_matrix")


class UnityInferSet3d(Dataset):

    def __init__(self,
                 filehash_list,
                 png_cache_dir,
                 image_crop_size=(640, 640),
                 image_out_size=(320, 320),
                 device="cpu",
                 name=None):

        super().__init__()

        self.filehash_list = filehash_list

        self.png_cache_dir = png_cache_dir
        self.image_crop_size = image_crop_size
        self.image_out_size = image_out_size
        self.name = name
        self.device = device
        self.rgb_convert = torch.tensor([], device=device, dtype=torch.float32)

    def __len__(self):
        return len(self.filehash_list)

    def __getitem__(self, idx):

        filehash = self.filehash_list[idx]

        image_crop_size = self.image_crop_size
        device = self.device

        unity_o = SecurionSource(unity_code=filehash, png_cache_dir=self.png_cache_dir)

        image_paths = unity_o.get_all_frames_path()
        images = []

        label_height_shift = 0
        label_width_shift = 0
        for image_path in image_paths:
            image, label_height_shift, label_width_shift = read_image_and_crop_into_tensor(image_path=image_path, image_crop_size=image_crop_size, device=device, name="main", shrink_to_image=True)
            if self.image_out_size != self.image_crop_size:
                image = image.to(torch.float32).unsqueeze(0)
                image = torch.nn.functional.interpolate(image, size=self.image_out_size, mode="bilinear")
                image = image.squeeze(0).to(torch.uint8)

            images.append(image[[0], ...])  # as we only can cope with b&w

        image = torch.cat(images).unsqueeze(0)

        return UnityInferSet3dT(image=image,
                                unity_i_code=unity_o.unity_i_code,
                                label_height_shift=label_height_shift,
                                label_width_shift=label_width_shift)


class UnityInferSet3dDICOM(Dataset):

    def __init__(self,
                 path_list,
                 image_crop_size=(640, 640),
                 image_out_size=(320, 320),
                 device="cpu",
                 name=None):

        super().__init__()

        self.path_list = path_list

        self.image_crop_size = image_crop_size
        self.image_out_size = image_out_size
        self.name = name
        self.device = device
        self.rgb_convert = torch.tensor([], device=device, dtype=torch.float32)

    def __len__(self):
        return len(self.path_list)

    def load_dicom(self, path):
        pass

    def __getitem__(self, idx):

        file_path, filehash = self.path_list[idx]

        image_crop_size = self.image_crop_size
        device = self.device

        unity_o = SecurionSource(unity_code=filehash, png_cache_dir=self.png_cache_dir)

        image_paths = unity_o.get_all_frames_path()
        images = []

        label_height_shift = 0
        label_width_shift = 0
        for image_path in image_paths:
            image, label_height_shift, label_width_shift = read_image_and_crop_into_tensor(image_path=image_path, image_crop_size=image_crop_size, device=device, name="main")
            if self.image_out_size != self.image_crop_size:
                image = image.to(torch.float32).unsqueeze(0)
                image = torch.nn.functional.interpolate(image, size=self.image_out_size, mode="bilinear")
                image = image.squeeze(0).to(torch.uint8)

            images.append(image[[0], ...])  # as we only can cope with b&w

        image = torch.cat(images).unsqueeze(0)

        return UnityInferSet3dT(image=image,
                                unity_i_code=unity_o.unity_i_code,
                                label_height_shift=label_height_shift,
                                label_width_shift=label_width_shift)



def load_echo_dicom(input_path, reject_color = False, reject_still = False):
    MATT_SOFT_PRIVATE_GROUP = 0x3551
    MATT_SOFT_PRIVATE_ELEMENT = 0x0077
    MATT_SOFT_PRIVATE_NAME = "Magiquant"
    MATT_SOFT_HASHNAME_ELEMENT_OFFSET = 0x01

    logger = logging.getLogger()

    try:
        dcm = pydicom.read_file(input_path)
        logger.info(f"{input_path} Opened")
    except Exception as e:
        logger.info(f"{input_path} Failed to open: {input_path}")
        raise Exception

    ###

    print("Hello")

    if reject_color:
        if dcm.get('UltrasoundColorDataPresent', None):
            logger("Rejection Color Data Present")
            raise Exception

    ###
    try:
        in_TransferSyntaxUID = dcm.file_meta.TransferSyntaxUID

        in_StudyInstanceUID = dcm.StudyInstanceUID
        in_SeriesInstanceUID = dcm.SeriesInstanceUID

        in_PhotometricInterpretation = dcm.PhotometricInterpretation

        InstanceNumber = int(dcm.InstanceNumber)
    except Exception as e:
        logger.warning(f"{input_path} Failed to extract key data: {str(e)}")
        raise Exception

    Modality = dcm.get('Modality', None)

    in_SOPClassUID = dcm.SOPClassUID

    PatientOrientation = dcm.get('PatientOrientation', None)
    SequenceOfUltrasoundRegions = dcm.get('SequenceOfUltrasoundRegions', None)

    Manufacturer = dcm.get('Manufacturer', None)
    ManufacturerModelName = dcm.get('ManufacturerModelName', None)

    in_SamplesPerPixel = dcm.SamplesPerPixel
    Rows = dcm.Rows
    Columns = dcm.Columns
    NumberOfFrames = dcm.get('NumberOfFrames', None)
    if NumberOfFrames is not None:
        NumberOfFrames = int(NumberOfFrames)

    if reject_still:
        if not NumberOfFrames:
            logger.warning("Rejection Still Frame")
            raise Exception

    WW = dcm.get("WindowWidth", None)
    WC = dcm.get("WindowCenter", None)

    if Modality in ["OCT"]:
        x0 = SequenceOfUltrasoundRegions[0].RegionLocationMinX0
        x1 = SequenceOfUltrasoundRegions[0].RegionLocationMaxX1
        y0 = SequenceOfUltrasoundRegions[0].RegionLocationMinY0
        y1 = SequenceOfUltrasoundRegions[0].RegionLocationMaxY1
        new_oct_rows = y1 - y0
        new_oct_cols = x1 - x0
        if new_oct_rows != new_oct_cols:
            raise Exception("OCT new_rows != new_cols")
        OCT_IMAGE_RESHAPE = True
    if Modality in ["IVOCT"]:
        x0 = 0
        x1 = dcm.pixel_array.shape[2]
        y0 = 0
        y1 = dcm.pixel_array.shape[1]
        new_oct_rows = y1 - y0
        new_oct_cols = x1 - x0
        OCT_IMAGE_RESHAPE = True
    else:
        OCT_IMAGE_RESHAPE = False


    FrameTime = dcm.get('FrameTime', None)
    ActualFrameDuration = dcm.get('ActualFrameDuration', None)
    RecommendedDisplayFrameRate = dcm.get('RecommendedDisplayFrameRate', None)
    FrameTimeVector = dcm.get('FrameTimeVector', None)
    if FrameTimeVector is not None:
        FrameTimeVector = list(FrameTimeVector)
    FrameIncrementPointer = dcm.get('FrameIncrementPointer', None)
    FrameDelay = dcm.get('FrameDelay', None)

    SOPInstanceUID = pydicom.uid.generate_uid()

    SeriesNumber = dcm.SeriesNumber

    SeriesDescription = dcm.get("SeriesDescription", None)
    if SeriesDescription is None:
        SeriesDescriptionText = ""
    else:
        SeriesDescriptionText = f"_{SeriesDescription}"

    PixelAspectRatio = dcm.get('PixelAspectRatio', None)

    LossyImageCompression = dcm.get('LossyImageCompression', None)

    UltrasoundColorDataPresent = dcm.get('UltrasoundColorDataPresent', None)

    ImageType = dcm.get("ImageType", None)
    if ImageType is None:
        print(f"{input_path}: No ImageType")
        raise Exception

    in_FileHash = dcm.get((MATT_SOFT_PRIVATE_GROUP, 0x0077), None)

    PatientSex = dcm.get("PatientSex", None)

    DOB = dcm.get("PatientBirthDate", None)
    if DOB is not None:
        PatientBirthYear = DOB[:4]
    else:
        PatientBirthYear = None

    SeriesDate = dcm.get("SeriesDate", None)
    if SeriesDate is not None:
        SeriesYear = SeriesDate[:4]
    else:
        SeriesYear = None

    ##

    if Modality not in ["US", "OCT", "IVOCT", "MR"]:
        logger.info(f"{input_path} Modality was not recognised - {Modality}")
        raise Exception

    ###

    if in_SOPClassUID == '1.2.840.10008.5.1.4.1.1.3': # Multi-frame
        SOPClassUID = '1.2.840.10008.5.1.4.1.1.3.1'
        sop_class = "us mf"
    elif in_SOPClassUID == '1.2.840.10008.5.1.4.1.1.3.1':
        SOPClassUID = '1.2.840.10008.5.1.4.1.1.3.1'
        sop_class = "us mf"
    elif in_SOPClassUID == '1.2.840.10008.5.1.4.1.1.6': # Single Frame
        SOPClassUID = '1.2.840.10008.5.1.4.1.1.6.1'
        sop_class = "us sf"
    elif in_SOPClassUID == '1.2.840.10008.5.1.4.1.1.6.1':
        SOPClassUID = '1.2.840.10008.5.1.4.1.1.6.1'
        sop_class = "us sf"
    elif in_SOPClassUID == '1.2.840.10008.5.1.4.1.1.14.1': #Multi-frame IVOCT
        SOPClassUID = '1.2.840.10008.5.1.4.1.1.14.1'
        sop_class = "ivoct mf"
    elif in_SOPClassUID == '1.2.840.10008.5.1.4.1.1.4':
        SOPClassUID = '1.2.840.10008.5.1.4.1.1.4'
        sop_class = "mri"
    else:
        logger.info(f'SOPClassUID {in_SOPClassUID} not in accepted US list')
        raise Exception

    ###

    try:
        if 'INVALID' in dcm.ImageType:
            logger.info(f"{input_path} Ignored - Report: {input_path}")
            # These files are reports with name burnt in a different place. So skip
            return
    except Exception as e:
        logger.info(f'{input_path} Ignored - No ImageType: {input_path}')
        raise Exception

    ###

    logging.warning(f"Processing Image Data")
    if in_TransferSyntaxUID == pydicom.uid.RLELossless:
        logger.info(f"{input_path} Input: RLE")
    elif in_TransferSyntaxUID == pydicom.uid.ImplicitVRLittleEndian:
        logger.info(f"{input_path} Input: Raw")
    elif in_TransferSyntaxUID == pydicom.uid.JPEGBaseline:
        logger.info(f"{input_path} Input: JPEG")
    elif in_TransferSyntaxUID == pydicom.uid.ExplicitVRLittleEndian:
        logger.info(f"{input_path} Input: ExplicitVRLittleEndian")
    else:
        logger.info(f"{input_path} Input: {in_TransferSyntaxUID}")

    ###

    logger.info(f"{input_path} In PhotometricInterpretation: {in_PhotometricInterpretation}")

    try:

        if in_PhotometricInterpretation == 'PALETTE COLOR':
            logger.info(f"{input_path} Applying LUT")
            pixel_array = pydicom.pixel_data_handlers.util.apply_color_lut(dcm.pixel_array, dcm)
            PhotometricInterpretation = "RGB"
            SamplesPerPixel = 3
        elif "YBR" in in_PhotometricInterpretation:
            pixel_array = pydicom.pixel_data_handlers.util.convert_color_space(dcm.pixel_array, dcm.PhotometricInterpretation, "RGB")
            PhotometricInterpretation = "RGB"
            SamplesPerPixel = 3
        elif in_PhotometricInterpretation == 'MONOCHROME1':
            pixel_array = dcm.pixel_array.copy()
            PhotometricInterpretation = "MONOCHROME1"
            SamplesPerPixel = 1
        elif in_PhotometricInterpretation == 'MONOCHROME2':
            if Modality == 'IVOCT':
                pixel_array = pydicom.pixel_data_handlers.util.apply_color_lut(dcm.pixel_array, dcm)
                PhotometricInterpretation = "RGB"
                SamplesPerPixel = 3
            else:
                pixel_array = dcm.pixel_array.copy()
                PhotometricInterpretation = "MONOCHROME2"
                SamplesPerPixel = 1
        elif in_PhotometricInterpretation == "RGB":
            pixel_array = dcm.pixel_array.copy()
            PhotometricInterpretation = "RGB"
            SamplesPerPixel = 3
        else:
            logger.error(f"Unrecognised source PhotometricInterpretation: {in_PhotometricInterpretation}")
            raise Exception

    except Exception as e:
        logger.warning(f"Error reading images from {input_path}")
        logger.warning(str(e))
        raise Exception

    logger.info(f"{input_path} PhotometricInterpretation out: {PhotometricInterpretation}")

    del dcm
    ###

    if pixel_array.dtype == np.uint16:
        print(f"16 bit data")

    if NumberOfFrames is None or NumberOfFrames == 1:
        if pixel_array.ndim == 3:
            pixel_array = np.expand_dims(pixel_array, 0)
        elif pixel_array.ndim == 2:
            pixel_array = np.expand_dims(pixel_array, 0)
            pixel_array = np.expand_dims(pixel_array, 3)
        else:
            logger.warning(f"Something went wrong with frames and ndim")
            raise Exception

    elif NumberOfFrames > 2:
        if pixel_array.ndim == 4:
            pixel_array = pixel_array
        elif pixel_array.ndim == 3:
            pixel_array = np.expand_dims(pixel_array, 3)
        else:
            logger.warning(f"Something went wrong with frames and ndim")
            raise Exception

    if OCT_IMAGE_RESHAPE:
        pass
    elif "vscan" in str(ManufacturerModelName).lower():
        pass
    else:
        if pixel_array.ndim == 4:
            pixel_array[:, :Rows // 10, :, :] = 0
        else:
            logger.warning(f"Failed to anonymise")
            return

    return pixel_array
