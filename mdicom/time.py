import logging
from typing import Optional
import pydicom


def work_out_frame_time(dcm: pydicom.FileDataset, default: Optional[float]=None):
    FrameTime = getattr(dcm, "FrameTime", None)
    RecommendedDisplayFrameRate = getattr(dcm, "RecommendedDisplayFrameRate", None)

    if RecommendedDisplayFrameRate is not None:
        rdfr_frame_time = 1000 / RecommendedDisplayFrameRate

    if FrameTime is not None and RecommendedDisplayFrameRate is None:
        logging.info(f"Using FrameTime {FrameTime}ms as RecommendedDisplayFrameRate is None")
        return FrameTime
    elif FrameTime is None and RecommendedDisplayFrameRate is not None:
        logging.info(f"Using RecommendedDisplayFrameRate {RecommendedDisplayFrameRate}fps as FrameTime is None")
        return rdfr_frame_time
    elif FrameTime is not None and RecommendedDisplayFrameRate is not None:
        if FrameTime > 0.9 * rdfr_frame_time and FrameTime < 1.1 * rdfr_frame_time:
            logging.info(f"FrameTime {FrameTime}ms and RecommendedDisplayFrameRate {RecommendedDisplayFrameRate}fps match")
            return FrameTime
        else:
            if FrameTime <= rdfr_frame_time:
                logging.info(f"Using FrameTime {FrameTime}ms as shorter than RecommendedDisplayFrameRate {RecommendedDisplayFrameRate}fps")
                return FrameTime
            else:
                logging.info(f"Using RecommendedDisplayFrameRate {RecommendedDisplayFrameRate}fps as shorter than FrameTime {FrameTime}ms")
                return rdfr_frame_time
    else:
        return default
