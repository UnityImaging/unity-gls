import io
import numpy as np
import pydicom
import polarTransform

# https://github.com/SlicerHeart/SlicerHeart/blob/d8f9b6e45ddc3b869046c151210f3dae0b80eb82/GeUsMovieReader/Logic/vtkSlicerGeUsMovieReaderLogic.cxx


def fix_gems_dicom(dcm: pydicom.Dataset, debug=False):
    if debug:
        import matplotlib
        matplotlib.use("MacOSX")
        import matplotlib.pyplot as plt

    targets = dcm["6003", "1010"]

    target_idx = None

    for idx, target in enumerate(targets):
        if target["0008", "2111"].value == "PrintPreview":
            target_idx = idx
            print(target_idx)

    if target_idx:
        target = targets[target_idx]
        target_SamplesPerPixel = target["0028", "0002"].value
        target_Rows = target["0028", "0010"].value
        target_Colums = target["0028", "0011"].value
        target_PixelArray = target["6003", "1011"].value

        target_ref_columns = None
        target_ref_rows = None
        for region in dcm.SequenceOfUltrasoundRegions:
            target_ref_columns = target_Colums // 2
            target_ref_rows = 1

        target_image_np = np.frombuffer(target_PixelArray, dtype=np.uint8).reshape(
            (target_Rows, target_Colums, target_SamplesPerPixel))
        target_np = target_image_np.copy()
        mask = np.std(target_np, axis=-1)
        mask = mask < 5
        mask = np.bitwise_not(mask)
        target_np[mask] = (255, 255, 255)
        target_np = (255, 255, 255) - target_np
        if debug:
            plt.imshow(target_np)
            plt.show()
        target_row_sum = target_np.sum(axis=2).sum(axis=0)
        right_col = np.where(target_row_sum[target_ref_columns:] == 0)[0][0] + target_ref_columns - 1
        left_col = np.where(target_row_sum[:target_ref_columns] == 0)[0][-1] + 1

        left_height = np.where(target_np[:, left_col, :].sum(-1) > 0)[0][-1]
        right_height = np.where(target_np[:, right_col, :].sum(-1) > 0)[0][-1]

        if debug:
            plt.imshow(target_np)
            plt.plot(target_np)
            plt.scatter(right_col, right_height)
            plt.scatter(left_col, left_height)
            plt.show()

        random_data = dcm["7fe1", "1001"][0]["7fe1", "1010"][0]["7fe1", "1020"][0]["7fe1", "1026"]
        polar_Rows = None
        polar_Columns = None
        for data in random_data:
            if ("7fe1", "1086") in data:
                polar_Rows = data["7fe1", "1086"][0]
                polar_Columns = data["7fe1", "1086"][1]

        if not polar_Rows and not polar_Columns:
            raise Exception("Cannot get polar size")

        segments = dcm["7fe1", "1001"][0]["7fe1", "1010"][0]["7fe1", "1020"][0]["7fe1", "1036"]
        polar_io = io.BytesIO()
        time_vector_io = io.BytesIO()
        frames = 0
        for segment in segments:
            frames = frames + segment["7fe1", "1037"].value
            polar_io.write(segment["7fe1", "1060"].value)
            time_vector_io.write(segment["7fe1", "1043"].value)

        time_vector_np = np.frombuffer(time_vector_io.getvalue(), dtype=np.float64)

        FrameTime = np.mean(np.diff(time_vector_np)) * 1000
        FrameTimeVector = [0]
        FrameTimeVector.extend(np.diff(time_vector_np) * 1000)

        # z(frames rather than color), theta, r (C format)
        required_data_len = frames * polar_Columns * polar_Rows
        polar_io.seek(0)
        polar_np = np.frombuffer(polar_io.getvalue(), count=required_data_len, dtype=np.uint8).reshape(
            (frames, polar_Columns, polar_Rows))

        # cartesian c is in z, y, x

        # center = ((right_col - left_col)//2, 0) # x,y
        center = (target_ref_columns, target_ref_rows)

        initialRadius = 0  # where the starting radius of image is
        finalRadius = int(
            np.sqrt(np.square(right_col - target_ref_columns) + np.square(right_height - target_ref_rows)))
        initialAngle = np.arctan2(right_height - target_ref_rows, right_col - target_ref_columns)
        finalAngle = np.arctan2(left_height - target_ref_rows, left_col - target_ref_columns)
        # imageSize = (finalRadius, right_col - left_col) # y,x
        imageSize = (target_Rows, target_Colums)
        hasCol = False

        cartesian_frames, tns = polarTransform.convertToCartesianImage(image=polar_np,
                                                                       center=center,
                                                                       initialRadius=initialRadius,
                                                                       finalRadius=finalRadius,
                                                                       initialAngle=initialAngle,
                                                                       finalAngle=finalAngle,
                                                                       imageSize=imageSize,
                                                                       useMultiThreading=True)

        if debug:
            plt.imshow(target_image_np)
            plt.show()
            plt.imshow(cartesian_frames[0, ...], cmap="gray")
            plt.show()

        match_np = target_image_np.copy()
        match_np[:, :, 0] = cartesian_frames[0, ...]
        if debug:
            plt.imshow(match_np)
            plt.show()

        SOPClassUID = '1.2.840.10008.5.1.4.1.1.3.1'
        SamplesPerPixel = 1
        BitsAllocated = 8
        BitsStored = 8
        HighBit = 7
        PhotometricInterpretation = "MONOCHROME2"
        TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        NumberOfFrames = frames

        dcm.file_meta.MediaStorageSOPClassUID = SOPClassUID
        dcm.SOPClassUID = SOPClassUID

        dcm.file_meta.TransferSyntaxUID = TransferSyntaxUID

        dcm.SamplesPerPixel = SamplesPerPixel
        dcm.NumberOfFrames = NumberOfFrames
        dcm.PhotometricInterpretation = PhotometricInterpretation

        dcm.BitsAllocated = BitsAllocated
        dcm.BitsStored = BitsStored
        dcm.HighBit = HighBit
        dcm.PixelRepresentation = 0
        if SamplesPerPixel > 1:
            dcm.PlanarConfiguration = 0

        dcm.FrameTime = FrameTime
        dcm.FrameTimeVector = FrameTimeVector
        dcm.FrameIncrementPointer = ("0018","1063")

        dcm.UltrasoundColorDataPresent = 0 # AS we don't know how to process the color doppler
        dcm.PixelData = cartesian_frames.tobytes()

